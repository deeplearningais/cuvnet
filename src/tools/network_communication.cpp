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


namespace cuvnet { namespace network_communication {

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

    server::server(const std::string& url, const std::string& prefix, const std::string key){
        m_impl.reset(new connection(url,prefix,key));
    }
    
    void server::push_merged(){

        for(std::map<std::string, bool>::iterator it = m_need_push.begin();
                it != m_need_push.end();
                it++){
            if(! it->second )
                continue;
            
            std::ostringstream os(std::ios::binary);
            {
                boost::archive::binary_oarchive oa(os);
                oa << m_merged[it->first];
            }
            std::string ms = os.str();
            mongo::BSONObjBuilder bob;
            bob << mongo::GENOID
                <<"key"<<m_impl->m_key
                <<"name"<< it->first
                <<"params"<< true
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
            m_merged[name] = m;
            m_need_push[name] = true;
        }

    }
    void server::merge(){

        mongo::BSONObj query = BSON(
                    "params" << true <<
                    "state" << "new" <<
                    "key" << m_impl->m_key),
            cmd = BSON(
                    "findAndModify" << "nc" <<
                    "query" << query <<
                    "update"<<BSON("$set"<<BSON("state"<<"staged")));

        while(true){
            mongo::BSONObj res;
            m_impl->m_con.runCommand(m_impl->m_prefix, cmd, res);
            CHECK_DB_ERR(m_impl->m_con);

            if(!res["value"].isABSONObj())
                break;

            mongo::BSONObj f = res["value"].Obj();

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
            if( m_merged.find(name) == m_merged.end() )
                m_merged[name] = m;
            else{
                cuv::apply_binary_functor(m_merged[name],m,cuv::BF_AXPBY, 0.5f, 0.5f);
            }

            m_need_push[name] = true;
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

    htensor_t client::fetch_merged(const std::string& s){
        mongo::BSONObj f = m_impl->m_con.findOne(m_impl->m_prefix+".nc",
                QUERY("name"<<s
                    <<"params"<<true
                    <<"state"<<"merged"
                    <<"key"<<m_impl->m_key));
        if(!f.hasField("name"))
            throw value_not_found_exception();

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
    void client::put_for_merging(const std::string& s, htensor_t& m){

        std::ostringstream os(std::ios::binary);
        {
            boost::archive::binary_oarchive oa(os);
            oa << m;
        }
        std::string ms = os.str();

        mongo::BSONObjBuilder bob;
        bob <<"key"<<m_impl->m_key
            <<"params"<< true
            <<"name"<< s
            <<"source"<< m_id
            <<"state"<<"new";
        bob.appendBinData("content",ms.size(),mongo::BinDataGeneral,&ms[0]);

        // upsert value
        m_impl->m_con.update( m_impl->m_prefix + ".nc", 
                BSON("key"<<m_impl->m_key
                    <<"params"<<true
                    <<"name"<<s
                    <<"source"<<m_id),
                bob.obj(), true);
        CHECK_DB_ERR(m_impl->m_con);
    }

    void server::cleanup(){
        m_impl->m_con.remove( m_impl->m_prefix+".nc",
                BSON("key" << m_impl->m_key));
    }
    void server::run(unsigned int sleep_sec, int n){
        pull_merged();
        for (int i = 0; i < n || n<0; ++i){
            merge();
            push_merged();
            boost::this_thread::sleep(boost::posix_time::seconds(1));
        }
    }

    void param_synchronizer::operator()(){
        if( ++m_cnt % m_push_steps == 0){
            int idx = 0;
            BOOST_FOREACH(Op* op, m_ops){
                ParameterInput* inp = dynamic_cast<ParameterInput*>(op);
                cuvAssert(inp);
                std::string name = inp->name() + boost::lexical_cast<std::string>(idx);
                htensor_t m = inp->data();
                m_client.put_for_merging(name, m);
                idx ++;
            }
        }
        if( ++m_cnt % m_pull_steps == 0){
            int idx = 0;
            BOOST_FOREACH(Op* op, m_ops){
                ParameterInput* inp = dynamic_cast<ParameterInput*>(op);
                cuvAssert(inp);
                std::string name = inp->name() + boost::lexical_cast<std::string>(idx);
                try{
                    matrix m = m_client.fetch_merged(name);
                    cuv::apply_binary_functor(inp->data(),m,cuv::BF_AXPBY, 0.5f, 0.5f);
                }catch(value_not_found_exception){
                }
                idx ++;
            }
        }
    }
        
} }
