#include <map>
#include <sstream>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/asio.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/bind.hpp>
#include <cuv/basics/io.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <mongo/client/dbclient.h>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/tools/gradient_descent.hpp> /* for network_stop exception */
#include <cuv/tools/timing.hpp>
#include <cuv/libs/opt/opt.hpp>
#include <cuv/tensor_ops/rprop.hpp>
#include <cuvnet/tools/logging.hpp>

#include "network_communication.hpp"

#ifdef NDEBUG
#  define CHECK_DB_ERR(CON)
#else
#  define CHECK_DB_ERR(CON)\
            {\
                std::string e = (CON).getLastError();\
                if(!e.empty()){\
                    throw std::runtime_error("nc: error_code!=0, failing: " + e + "\n" + (CON).getLastErrorDetailed().toString() );\
                }\
            }
#endif

namespace {
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("netcom"));
}

namespace cuvnet { namespace network_communication {

    void merger::add_param(const std::string& name, const htensor_t& t){
        m_merged[name] = t;
    }
    void merger::merge(const std::string& name, const htensor_t& delta){
        assert(has(name));
        cuv::learn_step_weight_decay( m_merged[name], delta, m_learnrate, 0.f);
    }
    bool merger::has(const std::string& name){
        return m_merged.find(name) != m_merged.end();
    }
    htensor_t& merger::operator[](const std::string& name){
        assert(has(name));
        return m_merged[name];
    }

    momentum_merger::momentum_merger(float learnrate, float momentum)
        :merger(learnrate), m_momentum(momentum){}
    void momentum_merger::add_param(const std::string& name, const htensor_t& t){
        m_merged[name] = t;
        m_moments[name] = htensor_t(t.shape());
        m_moments[name] = 0.f;
    }
    void momentum_merger::merge(const std::string& name, const htensor_t& delta){
        assert(has(name));
        assert(m_moments.find(name) != m_moments.end());
        htensor_t& m = m_moments[name];
        cuv::apply_binary_functor(m, delta, cuv::BF_AXPY, m_momentum);
        cuv::learn_step_weight_decay( m_merged[name], m, m_learnrate, 0.f);
        //m_merged[name] += m;
    }

    adagrad_merger::adagrad_merger(float learnrate, float delta, int winsize)
        :merger(learnrate), m_delta(delta), m_winsize(winsize){}
    void adagrad_merger::add_param(const std::string& name, const htensor_t& t){
        m_merged[name] = t;
        m_sq_grad_sum[name] = htensor_t(t.shape());
        m_sq_grad_sum[name] = 0.f;
        m_count[name] = 0;
    }
    void adagrad_merger::merge(const std::string& name, const htensor_t& delta){
        assert(has(name));
        assert(m_sq_grad_sum.find(name) != m_sq_grad_sum.end());
        htensor_t& sqsums = m_sq_grad_sum[name];

        htensor_t s = delta*delta;
        sqsums += s;

        // assume that the learning rate was already applied to lr in the worker!
        cuv::libs::opt::adagrad(m_merged[name], delta, sqsums, m_learnrate, m_delta, 0.f, 0.f);

        if(++m_count[name] % m_winsize == 0)
        {
            LOG4CXX_INFO(g_log, "adagrad_merger: Resetting sums for "<<name);
            sqsums = 0.f;
        }
    }
    

    struct connection{
        connection(const std::string& url, const std::string& prefix, const std::string key){
            m_con.connect(url);
            m_con.createCollection(prefix+".nc");
            m_key = key;
            m_prefix = prefix;
        }

        std::string m_prefix;
        std::string m_key;
        mongo::DBClientConnection m_con;
    };

    server::server(const std::string& url, const std::string& prefix, const std::string key, merger* m){
        m_stop = false;
        m_impl.reset(new connection(url,prefix,key));
        if(!m)
            throw std::runtime_error("server did not get a merger!");
        m_merger = m;
    }
    
    void server::push_merged(){

        for(std::map<std::string, bool>::iterator it = m_need_push.begin();
                it != m_need_push.end();
                it++){
            if(! it->second )
                continue;
            
            std::ostringstream os(std::ios::binary);
            {   // serialize the contents of the merged parameter
                assert(m_merger->has(it->first));
                boost::archive::binary_oarchive oa(os);
                oa << (*m_merger)[it->first];
            }
            std::string ms = os.str();
            mongo::BSONObjBuilder bob;
            bob <<"key"<<m_impl->m_key
                <<"name"<< it->first
                <<"params"<< true
                <<"version"<< m_versions[it->first]
                <<"state"<<"merged";
            bob.appendBinData("content",ms.size(),mongo::BinDataGeneral,&ms[0]);

            // upsert merged value
            m_impl->m_con.update( m_impl->m_prefix + ".nc",
                    BSON("key"<<m_impl->m_key
                        <<"name"<< it->first
                        <<"params"<< true
                        <<"state"<< "merged"),
                    bob.obj(), true);

            it->second = false;
        }

        // delete all items that contributed to the merged values
        m_impl->m_con.remove( m_impl->m_prefix+".nc",
                BSON("state" << "staged" << "key" << m_impl->m_key));
    }

    void server::pull_merged(){
        std::auto_ptr<mongo::DBClientCursor> p =
            m_impl->m_con.query( m_impl->m_prefix+".nc", 
                    QUERY(
                        "params" << true <<
                        "state" << "merged" <<
                        "key" << m_impl->m_key),
                    0,0);
        CHECK_DB_ERR(m_impl->m_con);
        while(p->more()){
            mongo::BSONObj f = p->next();
            std::string name = f["name"].String();
            if(m_merger->has(name)){
                // TODO: no need to pull this in the first place?
                continue;
            }

            htensor_t m;
            {
                int len;
                const char* p = f["content"].binData(len);
                std::string content(p, len);
                std::istringstream is(content, std::ios::binary);
                boost::archive::binary_iarchive ia(is);
                ia >> m;
            }


            m_versions[name] = f["version"].Int();
            m_merger->add_param(name, m);
            m_need_push[name] = false;
            LOG4CXX_WARN(g_log, "New parameter found: " << name << " norm: "<<cuv::norm2(m));
        }

    }
    void server::merge(){
        mongo::BSONObj query = BSON(
                "params" << true <<
                "state" << "delta" <<
                "key" << m_impl->m_key),
            cmd = BSON(
                    "findAndModify" << "nc" <<
                    "query" << query <<
                    "sort" << BSON("delta_idx"<<-1) <<
                    "update"<<BSON("$set"<<BSON("state"<<"staged")));

        int queue_len = m_impl->m_con.count(m_impl->m_prefix+".nc", query);
        CHECK_DB_ERR(m_impl->m_con);
        static int warned = 0;
        static const int warn_step = 400;
        if(queue_len < warned)
            warned -= warn_step;
        if(queue_len > warned + warn_step){
            //std::cout << "WARNING: Async SGD: queue-length is " << queue_len <<" trying to catch up..."<<std::endl;
            warned += warn_step;
        }

        int process_max = std::max((size_t) 5, 2*m_merger->n_params());

        std::auto_ptr<mongo::DBClientCursor> p =
            m_impl->m_con.query( m_impl->m_prefix+".nc",
                    QUERY("params"<<true
                       <<"state"<<"delta"
                       <<"key"<<m_impl->m_key).sort(BSON("delta_idx"<<-1)),process_max);
        CHECK_DB_ERR(m_impl->m_con);
        mongo::BSONArrayBuilder done_ids;

        for (int dummy = 0; dummy < process_max; ++dummy)
        {
/*
 *            mongo::BSONObj res;
 *            m_impl->m_con.runCommand(m_impl->m_prefix, cmd, res);
 *            CHECK_DB_ERR(m_impl->m_con);
 *
 *            if(!res["value"].isABSONObj())
 *                break;
 *
 *          mongo::BSONObj f = res["value"].Obj();
 */
            if(!p->more())
                break;
            mongo::BSONObj f = p->next();
            if(!f.hasField("name"))
                break;

            htensor_t m;
            {
                int len;
                const char* p = f["content"].binData(len);
                std::string content(p, len);
                std::istringstream is(content, std::ios::binary);
                boost::archive::binary_iarchive ia(is);
                ia >> m;
            }

            std::string name = f["name"].String();
            if( !m_merger->has(name) )
            {
                pull_merged();
                if( !m_merger->has(name) ){
                    throw value_not_found_exception();
                }
            }
            //LOG4CXX_WARN(g_log, "Merging: " << name << " norm: " << cuv::norm2(m));
            m_merger->merge(name, m);

            m_versions[name] ++;
            m_need_push[name] = true;
            done_ids << f["_id"];
        }
        m_impl->m_con.remove( m_impl->m_prefix+".nc", BSON("_id" << BSON("$in" << done_ids.arr())));
        CHECK_DB_ERR(m_impl->m_con);

        if(queue_len > warn_step)
        {
            LOG4CXX_WARN(g_log, "WARNING: Async SGD: queue-length is " << queue_len <<" --> clearing, you're loosing work!");
            m_impl->m_con.remove(m_impl->m_prefix+".nc", query);
        }

    }

    client::client(const std::string& url, const std::string& prefix, const std::string key, const std::string id){
        m_impl.reset(new connection(url,prefix,key));

        if(id.size()==0){
            std::string hostname(256, '\0');
            gethostname(&hostname[0], 256);
            m_id = (boost::format("%s:%d") % &hostname[0] % getpid()).str();
        }else{
            m_id = id;
        }
    }

    void client::send_stop_signal(const std::string& stage){
        m_impl->m_con.insert(m_impl->m_prefix+".nc", 
                BSON("key"<<m_impl->m_key
                    <<"signal"<<"stop"
                    <<"stage"<<stage));
    }
    bool client::got_stop_signal(const std::string& stage){
        mongo::BSONObj f = m_impl->m_con.findOne(m_impl->m_prefix+".nc",
                QUERY("key"<<m_impl->m_key
                    <<"signal"<<"stop"
                    <<"stage"<<stage));
        return f.hasField("key");
    }
    htensor_t client::fetch_merged(const std::string& s){
        mongo::BSONObj f = m_impl->m_con.findOne(m_impl->m_prefix+".nc",
                QUERY("name"<<s
                    <<"params"<<true
                    <<"state"<<"merged"
                    <<"version"<< mongo::GT <<m_versions[s]
                    <<"key"<<m_impl->m_key));
        if(!f.hasField("name"))
            throw value_not_found_exception();

        m_versions[s] = f["version"].Int();

        // ensure that the version we're sending isn't outdated
        // (if versions are simply counted up by some clients, then
        // some clients (the ones that joined later) are always preferred.
        m_delta_versions[s] = std::max(m_delta_versions[s], m_versions[s]);

        htensor_t m;
        {
            int len;
            const char* p = f["content"].binData(len);
            std::string content(p, len);
            std::istringstream is(content, std::ios::binary);
            boost::archive::binary_iarchive ia(is);
            ia >> m;
        }
        return m;
    }
    void client::put_for_merging(const std::string& s, const htensor_t& delta, const matrix& current_value){

        int cnt = 
            m_impl->m_con.count( m_impl->m_prefix + ".nc",
                    BSON("key"<<m_impl->m_key
                        <<"params"<<true
                        <<"name"<<s
                        <<"state"<<"merged"));
        if(cnt == 0){
            // we need to put a weight obj on the server in the 1st place...
            std::ostringstream os(std::ios::binary);
            {   // serialize
                boost::archive::binary_oarchive oa(os);
                oa << current_value;
            }
            std::string ms = os.str();

            mongo::BSONObjBuilder bob;
            bob <<"key"<<m_impl->m_key
                <<"params"<< true
                <<"name"<< s
                <<"version"<< 0
                <<"state"<<"merged"; // pretend it is "merged"
            bob.appendBinData("content",ms.size(),mongo::BinDataGeneral,&ms[0]);

            // upload object to the server
            LOG4CXX_WARN(g_log, "Uploading initial: " << s << " norm: " << cuv::norm2(current_value));
            m_impl->m_con.insert( m_impl->m_prefix + ".nc", bob.obj());
            CHECK_DB_ERR(m_impl->m_con);
        }

        {
            // we just need to send a weight /update/ to the server
            std::ostringstream os(std::ios::binary);
            {   // serialize
                boost::archive::binary_oarchive oa(os);
                oa << delta;
            }
            std::string ms = os.str();
            //LOG4CXX_WARN(g_log, "Uploading delta: " << s << " norm: " << cuv::norm2(delta));

            mongo::BSONObjBuilder bob;
            bob <<"key"<<m_impl->m_key
                <<"params"<< true
                <<"name"<< s
                <<"delta_idx"<< ++m_delta_versions[s]
                <<"source"<< m_id
                <<"state"<<"delta";
            bob.appendBinData("content",ms.size(),mongo::BinDataGeneral,&ms[0]);

            /*
             * upsert value 
             * NOTE: THIS DOES NOT work. updates are lost.
             * the clients set their accumulated gradient to 0
             * the update we're pushing now is not necessarily
             * merged by the server before we upsert again.
             *
             *m_impl->m_con.update( m_impl->m_prefix + ".nc", 
             *        BSON("key"<<m_impl->m_key
             *            <<"params"<<true
             *            <<"name"<<s
             *            <<"source"<<m_id),
             *        bob.obj(), true);
             */
            m_impl->m_con.insert( m_impl->m_prefix + ".nc", bob.obj());
            CHECK_DB_ERR(m_impl->m_con);
        }
    }

    void server::cleanup(){
        m_impl->m_con.remove( m_impl->m_prefix+".nc",
                BSON("key" << m_impl->m_key));
        m_impl->m_con.dropIndexes(m_impl->m_prefix+".nc");
        m_impl->m_con.ensureIndex(m_impl->m_prefix+".nc", 
                BSON("key"<<1<<"name"<<1));
        m_impl->m_con.ensureIndex(m_impl->m_prefix+".nc", 
                BSON("params"<<1<<"state"<<1<<"key"<<1<<"delta_idx"<<-1));
        //m_impl->m_con.ensureIndex(m_impl->m_prefix+".nc", BSON("delta_idx"<<1));
        m_impl->m_con.reIndex(    m_impl->m_prefix+".nc");
    }
    void server::run(unsigned int sleep_msec, int n){
        //pull_merged(); // merge does pull when delta for unknown obj is found
        for (int i = 0; (i < n || n<0) && !m_stop; ++i){
            merge();
            push_merged();
            boost::this_thread::sleep(boost::posix_time::milliseconds(sleep_msec));
        }
    }

    void param_synchronizer::test_stop(){
        if(m_client.got_stop_signal(m_stage))
        {
            LOG4CXX_INFO(g_log, "got stop signal from co-worker: "<< m_stage);
            throw network_stop();
        }
    }
    void param_synchronizer::stop_coworkers(){
        LOG4CXX_INFO(g_log, "sending stop to co-workers: "<< m_stage);
        m_client.send_stop_signal(m_stage);
    }
    void param_synchronizer::operator()(
            std::map<Op*, cuv::tensor<float, cuv::host_memory_space> >* updates,
            unsigned int, unsigned int
            ){

        // first try to pull!
        assert(m_pull_steps > m_pull_off);
        if( m_cnt % m_pull_steps == m_pull_off){
            int idx = 0;
            BOOST_FOREACH(Op* op, m_ops){
                ParameterInput* inp = dynamic_cast<ParameterInput*>(op);
                assert(inp);
                std::string name = m_stage + inp->name() + boost::lexical_cast<std::string>(idx);
                try{
                    matrix m = m_client.fetch_merged(name);
                    inp->data() = m;
                    //LOG4CXX_INFO(g_log, m_client.id() << ": got new value for "<<name << " norm: "<<cuv::norm2(m));
                }catch(value_not_found_exception){
                }
                idx ++;
            }
            m_pull_off = m_pull_steps / 2 + drand48() * (m_pull_steps/2);
        }

        ++m_cnt;

        // if this is the first time, we may have pulled new params, 
        // don't push anything, instead delete gradients!
        if(m_cnt == 1){
            BOOST_FOREACH(Op* op, m_ops){
                ParameterInput* inp = dynamic_cast<ParameterInput*>(op);
                assert(inp);
                std::map<Op*, cuv::tensor<float, cuv::host_memory_space> >::iterator 
                    it = updates->find(inp);
                if(it != updates->end())
                    it->second = 0.f;
            }
        }

        assert(m_push_steps > m_push_off);
        if( m_cnt % m_push_steps == m_push_off){
            int idx = 0;
            BOOST_FOREACH(Op* op, m_ops){
                ParameterInput* inp = dynamic_cast<ParameterInput*>(op);
                assert(inp);
                std::string name = m_stage + inp->name() + boost::lexical_cast<std::string>(idx);
                //htensor_t m = inp->data();
                std::map<Op*, cuv::tensor<float, cuv::host_memory_space> >::iterator 
                    it = updates->find(inp);
                if(it != updates->end()){
                    // push results to server
                    // we need to ensure that all updates are weighted equally here
                    // since they may incorporate more or less client updates
                    //it->second /= (float)(m_cnt-m_last_push);
                    m_client.put_for_merging(name, it->second, inp->data());
                    // now that results are pushed, start recording changes over
                    it->second = 0.f; 
                }
                idx ++;
            }
            m_last_push = m_cnt;
            m_push_off = m_push_steps / 2 + drand48() * (m_push_steps/2);
        }
    }
        
} }
