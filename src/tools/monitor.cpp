#include <map>
#include <iomanip>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <cuv/tools/device_tools.hpp>

#include <iostream>
#include <fstream>


#include "monitor.hpp"
namespace cuvnet
{
    struct monitor_impl{
        std::vector<watchpoint*> m_watchpoints; ///< keeps all watchpoints 
        std::map<std::string, watchpoint*> m_wpmap;  ///< access watchpoints by name
    };

    typedef
        boost::accumulators::accumulator_set<double,
        boost::accumulators::stats<
            boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::variance(boost::accumulators::lazy) > > acc_t;

    /// a catch-all representation of a watch point
    struct watchpoint{
        /** ctor
         * @param _t  the type of the watchpoint (e.g. scalar stats or just value sink)
         * @param _op the op we want to gather information about
         * @param _name a name for this watchpoint
         * @param result the number of the result of _op
         */
        watchpoint(monitor::watchpoint_type _t, boost::shared_ptr<Op> _op, const std::string& _name, unsigned int result=0)
            :type(_t),op(_op),name(_name)
        {
            switch(type){
                case monitor::WP_SINK:
                case monitor::WP_SCALAR_EPOCH_STATS:
                    sink.reset(new Sink(name,op->result(result)));
                    break;
                case monitor::WP_D_SINK:
                case monitor::WP_D_SCALAR_EPOCH_STATS:
                    dsink = delta_sink(name, op, result);
                    break;
                case monitor::WP_FUNC_SINK:
                case monitor::WP_FUNC_SCALAR_EPOCH_STATS:
                    func.reset(op,result,name);
                    break;
            }
        }
        monitor::watchpoint_type         type;
        function                func;
        boost::shared_ptr<Op>     op;
        std::string             name;
        boost::shared_ptr<DeltaSink> dsink;  ///< this sink is managed by the monitor itself
        boost::shared_ptr<Sink> sink;  ///< this sink is managed by the monitor itself
        acc_t           scalar_stats;
    };

    monitor::monitor(bool verbose, const std::string& file_name)
    :m_batch_presentations(0)
    ,m_epochs(0)
    ,m_cv_mode(CM_TRAIN)
    ,m_split(0)
    ,m_logfile(file_name.c_str(), std::ios::out | std::ios::app)
    ,need_header_log_file(true)
    ,m_verbose(verbose){
        m_impl.reset(new monitor_impl());
    }

    monitor::~monitor(){
        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            delete p;
        }
    }
    monitor& monitor::add(watchpoint_type type, boost::shared_ptr<Op> op, const std::string& name, unsigned int result){
        m_impl->m_watchpoints.push_back(new watchpoint(type,op,name,result));
        m_impl->m_wpmap[name] = m_impl->m_watchpoints.back();
        return *this;
    }
    void monitor::set_training_phase(cv_mode mode, int split){
        if(split != m_split || mode == CM_TRAINALL)
            m_epochs = 0;
        m_cv_mode = mode;
        m_split = split;
    }
    void monitor::after_batch(){
        m_batch_presentations ++;

        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            if(p->type == WP_SCALAR_EPOCH_STATS)
            {
                if(p->sink->cdata().size()==1)
                    p->scalar_stats((float)p->sink->cdata()[0]);
                else {
                    // TODO: need real stats, not this!
                    p->scalar_stats(cuv::maximum(p->sink->cdata()));
                    p->scalar_stats(cuv::mean(p->sink->cdata()));
                    p->scalar_stats(cuv::minimum(p->sink->cdata()));
                }
                p->sink->forget();
            }
            if(p->type == WP_D_SCALAR_EPOCH_STATS)
            {
                if(p->dsink->cdata().size()==1)
                    p->scalar_stats((float)p->dsink->cdata()[0]);
                else {
                    // TODO: need real stats, not this!
                    p->scalar_stats(cuv::maximum(p->dsink->cdata()));
                    p->scalar_stats(cuv::mean(p->dsink->cdata()));
                    p->scalar_stats(cuv::minimum(p->dsink->cdata()));
                }
                p->dsink->forget();
            }
            if(p->type == WP_FUNC_SCALAR_EPOCH_STATS)
            {
                p->scalar_stats((float)p->func.evaluate()[0]);
            }
        }
    }
    void monitor::before_epoch(){
        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            switch(p->type){
                case WP_SCALAR_EPOCH_STATS:
                case WP_FUNC_SCALAR_EPOCH_STATS:
                    p->scalar_stats = acc_t();
                    break;
                default:
                    break;
            }
        }

    }
    void monitor::after_epoch(){
        if(m_cv_mode == CM_TRAIN || m_cv_mode == CM_TRAINALL)
            m_epochs ++;
        if(m_verbose)
            simple_logging();
        if(m_logfile.is_open())
            log_to_file();
        m_batch_presentations = 0;
    }
    watchpoint& monitor::get(const std::string& name){
        std::map<std::string, watchpoint*>::iterator it 
            = m_impl->m_wpmap.find(name);
        if(it != m_impl->m_wpmap.end())
            return *it->second;
        throw std::runtime_error("Unknown watchpoint `"+name+"'");
    }
    const watchpoint& monitor::get(const std::string& name)const{
        std::map<std::string, watchpoint*>::const_iterator it 
            = m_impl->m_wpmap.find(name);
        if(it != m_impl->m_wpmap.end())
            return *it->second;
        throw std::runtime_error("Unknown watchpoint `"+name+"'");
    }
    float monitor::count(const std::string& name)const{
        return boost::accumulators::count(get(name).scalar_stats);
    }
    float monitor::mean(const std::string& name)const{
        return boost::accumulators::mean(get(name).scalar_stats);
    }
    float monitor::var(const std::string& name)const{
        return boost::accumulators::variance(get(name).scalar_stats);
    }
    float monitor::stddev(const std::string& name)const{
        return std::sqrt(boost::accumulators::variance(get(name).scalar_stats));
    }
    const matrix& monitor::operator[](const std::string& name){
        watchpoint& wp = get(name);
        switch(wp.type){
            case WP_SINK:
            case WP_SCALAR_EPOCH_STATS:
                return wp.sink->cdata();
            case WP_D_SINK:
            case WP_D_SCALAR_EPOCH_STATS:
                return wp.dsink->cdata();
            case WP_FUNC_SINK:
            case WP_FUNC_SCALAR_EPOCH_STATS:
                return wp.func.evaluate();
        }
        throw std::runtime_error("unknown watchpoint type");
    }
    const matrix& monitor::operator[](const boost::shared_ptr<Op>& op){
        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            if(p->op == op)
                return p->sink->cdata();
        }
        throw std::runtime_error("Unknown watchpoint requested");
    }
    void monitor::log_to_file(){
        typedef std::pair<std::string, std::string> ss_t;
        assert(m_logfile.is_open());
        if(need_header_log_file){
            m_logfile << "mode\tsplit\tepoch\tmem";
            need_header_log_file = false;
            BOOST_FOREACH(const ss_t& p, m_constants){
                m_logfile << '\t' << p.first;
            }
            BOOST_FOREACH(const watchpoint* p, m_impl->m_watchpoints){
                m_logfile << '\t' << p->name;
            }
            m_logfile << std::endl;
        }

        m_logfile << m_cv_mode << '\t' << m_split << '\t' << m_epochs << '\t' << cuv::getFreeDeviceMemory();
        BOOST_FOREACH(const ss_t& p, m_constants){
            m_logfile << '\t' << p.second;
        }
        BOOST_FOREACH(const watchpoint* p, m_impl->m_watchpoints){
            if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS){
                // writes to the file the loss
                m_logfile  << '\t' <<  mean(p->name)  << '\t' << stddev(p->name);
            }
        }
        m_logfile << std::endl;
    }
    void monitor::simple_logging()const{
        std::cout << "\r epoch "<<m_epochs<<":"<<m_batch_presentations<<",  free_mb="<<cuv::getFreeDeviceMemory()/1024/1024<<",  ";
        typedef std::pair<std::string, std::string> ss_t;
        BOOST_FOREACH(const ss_t& p, m_constants){
            std::cout << p.first<<"="<<p.second<<" ";
        }
        BOOST_FOREACH(const watchpoint* p, m_impl->m_watchpoints){
            if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS){
                std::cout  << p->name<<"="
                    << std::left << std::setprecision(4) << mean(p->name) <<" ("
                    << std::left << std::setprecision(4) << stddev(p->name)<<"),  ";
            }
        }
        std::cout << "           " << std::flush;
    }
}
