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

#include <log4cxx/logger.h>
#include <log4cxx/mdc.h>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/gradient_descent.hpp>

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
                case monitor::WP_SINK_ONCE_STATS:
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
                case monitor::WP_FULL_WEIGHT_STATS:
                case monitor::WP_CONV_WEIGHT_STATS:
                    // this is only done at end of epoch
                    break;
            }
        }
        inline void notify_scalar_stats(double d, cv_mode mode){
            switch(mode){
                case CM_TRAIN:
                case CM_TRAINALL:
                    scalar_stats(d);
                default:
                    scalar_stats_valid(d);
            };
        }
        inline const acc_t& stats(cv_mode mode)const{
            switch(mode){
                case CM_TRAIN:
                case CM_TRAINALL:
                    return scalar_stats;
                default:
                    return scalar_stats_valid;
            };
        }
        inline acc_t& stats(cv_mode mode){
            switch(mode){
                case CM_TRAIN:
                case CM_TRAINALL:
                    return scalar_stats;
                default:
                    return scalar_stats_valid;
            };
        }
        inline void reset(cv_mode mode){
            switch(mode){
                case CM_TRAIN:
                case CM_TRAINALL:
                    scalar_stats = acc_t();
                default:
                    scalar_stats_valid = acc_t();
            };
        }
        monitor::watchpoint_type         type;
        function                func;
        boost::shared_ptr<Op>     op;
        std::string             name;
        boost::shared_ptr<DeltaSink> dsink;  ///< this sink is managed by the monitor itself
        boost::shared_ptr<Sink> sink;  ///< this sink is managed by the monitor itself
        private:
            acc_t           scalar_stats;
            acc_t           scalar_stats_valid;
    };

    monitor::monitor(bool verbose, const std::string& file_name)
    :m_batch_presentations(0)
    ,m_every(0)
    ,m_epochs(0)
    ,m_cv_mode(CM_TRAIN)
    ,m_split(0)
    ,m_logfile(file_name.c_str(), std::ios::out | std::ios::app)
    ,need_header_log_file(true)
    ,m_verbose(verbose)
    {
        m_impl.reset(new monitor_impl());
    }

    monitor::~monitor(){
        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            delete p;
        }
    }
    monitor& monitor::add(watchpoint_type type, boost::shared_ptr<Op> op, const std::string& name, unsigned int result){
        if(op){
            op->result(result)->need_result = true;
            m_impl->m_watchpoints.push_back(new watchpoint(type,op,name,result));
            m_impl->m_wpmap[name] = m_impl->m_watchpoints.back();
        }
        return *this;
    }
    monitor& monitor::remove(const std::string& name){
        for(auto it = m_impl->m_watchpoints.begin();
                it != m_impl->m_watchpoints.end();
                ++it){
            if((*it)->name == name){
                m_impl->m_watchpoints.erase(it);
                break;
            }
        }
        auto it = m_impl->m_wpmap.find(name);
        if(it != m_impl->m_wpmap.end()){
            m_impl->m_wpmap.erase(it);
        }
        return *this;
    }
    
    void monitor::set_training_phase(cv_mode mode, int split){
        if(split != m_split || mode == CM_TRAINALL)
            m_epochs = 0;
        m_cv_mode = mode;
        m_split = split;
    }
    void update_stats(acc_t& acc, const matrix& mat){
        cuv::tensor<matrix::value_type, cuv::host_memory_space> v = mat;
        matrix::value_type* ptr = v.ptr();
        matrix::value_type* end = ptr + v.size();
        acc = acc_t();
        for(; ptr != end; ptr++)
            acc(*ptr);
    }
    void monitor::after_batch(unsigned int epoch, unsigned int bid){
        if((m_cv_mode == CM_TRAIN || m_cv_mode == CM_TRAINALL) && m_every > 0 && m_batch_presentations % m_every == 0){
            // we have logged everything in the previous step
            reset();
        }
        // TODO reset this to value before eval epoch???
        if(m_cv_mode == CM_TRAIN || m_cv_mode == CM_TRAINALL)
            m_batch_presentations ++;

        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            if(p->type == WP_SCALAR_EPOCH_STATS)
            {
                if(p->sink->cdata().size()==1){
                    float f = (float)p->sink->cdata()[0];
                    if(!std::isfinite(f)){
                        throw arithmetic_error_stop();
                    }
                    p->notify_scalar_stats(f, m_cv_mode);
                }else {
                    // TODO: need real stats, not this!
                    p->notify_scalar_stats(cuv::maximum(p->sink->cdata()), m_cv_mode);
                    p->notify_scalar_stats(cuv::mean(p->sink->cdata()), m_cv_mode);
                    p->notify_scalar_stats(cuv::minimum(p->sink->cdata()), m_cv_mode);
                }
                p->sink->forget();
            }
            if(p->type == WP_D_SCALAR_EPOCH_STATS)
            {
                if(p->dsink->cdata().size()==1)
                    p->notify_scalar_stats((float)p->dsink->cdata()[0], m_cv_mode);
                else {
                    // TODO: need real stats, not this!
                    p->notify_scalar_stats(cuv::maximum(p->dsink->cdata()), m_cv_mode);
                    p->notify_scalar_stats(cuv::mean(p->dsink->cdata()), m_cv_mode);
                    p->notify_scalar_stats(cuv::minimum(p->dsink->cdata()), m_cv_mode);
                }
                p->dsink->forget();
            }
            if(p->type == WP_FUNC_SCALAR_EPOCH_STATS)
            {
                p->notify_scalar_stats((float)p->func.evaluate()[0], m_cv_mode);
            }
            if(p->type == WP_SINK_ONCE_STATS && bid == 0)
            {
                update_stats(p->stats(m_cv_mode), p->sink->cdata());
            }
        }
        if((m_cv_mode == CM_TRAIN || m_cv_mode == CM_TRAINALL) && m_every > 0 && m_batch_presentations % m_every == 0)
            log();
    }
    void monitor::log(){
        if(m_verbose)
            simple_logging();
        standard_logging();
        if(m_logfile.is_open())
            log_to_file();
    }
    void monitor::reset(){
        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            p->reset(m_cv_mode);
            switch(p->type){
                case WP_FULL_WEIGHT_STATS:
                case WP_CONV_WEIGHT_STATS:
                    update_stats(p->stats(m_cv_mode), boost::dynamic_pointer_cast<ParameterInput>(p->op)->data());
                    break;
                    //case WP_SINK_ONCE_STATS:
                    //    update_stats(p->scalar_stats, p->sink->cdata());
                    //    break;
                default:
                    break;
            }
        }
    }
    void monitor::after_epoch(){
        if(m_cv_mode == CM_TRAIN || m_cv_mode == CM_TRAINALL)
            m_epochs ++;

        if(m_cv_mode == CM_VALID || m_every == 0)
            log();
    }
    bool monitor::has(const std::string& name)const{
        std::map<std::string, watchpoint*>::const_iterator it 
            = m_impl->m_wpmap.find(name);
        return it != m_impl->m_wpmap.end();
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
        return boost::accumulators::count(get(name).stats(m_cv_mode));
    }
    float monitor::mean(const std::string& name)const{
        return boost::accumulators::mean(get(name).stats(m_cv_mode));
    }
    float monitor::var(const std::string& name)const{
        return boost::accumulators::variance(get(name).stats(m_cv_mode));
    }
    float monitor::stddev(const std::string& name)const{
        return std::sqrt(boost::accumulators::variance(get(name).stats(m_cv_mode)));
    }
    const matrix& monitor::operator[](const std::string& name){
        watchpoint& wp = get(name);
        switch(wp.type){
            case WP_FULL_WEIGHT_STATS:
            case WP_CONV_WEIGHT_STATS:
                return boost::dynamic_pointer_cast<ParameterInput>(wp.op)->data();
            case WP_SINK:
            case WP_SINK_ONCE_STATS:
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
            if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS || p->type == WP_FULL_WEIGHT_STATS  || p->type == WP_CONV_WEIGHT_STATS){
                m_logfile  << '\t' <<  mean(p->name);
            }
        }
        m_logfile << std::endl;
    }
    void monitor::standard_logging()const{
        log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("mon"));


        std::vector<log4cxx::MDC*> v;
        BOOST_FOREACH(const watchpoint* p, m_impl->m_watchpoints){
            if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS || p->type == WP_FULL_WEIGHT_STATS  || p->type == WP_CONV_WEIGHT_STATS || p->type == WP_SINK_ONCE_STATS){
                log4cxx::MDC* mdc0 = new log4cxx::MDC(p->name + "_mean", boost::lexical_cast<std::string>(mean(p->name)));
                v.push_back(mdc0);
                log4cxx::MDC* mdc1 = new log4cxx::MDC(p->name + "_var", boost::lexical_cast<std::string>(var(p->name)));
                v.push_back(mdc1);
            }
        }
        typedef std::pair<std::string, std::string> ss_t;
        BOOST_FOREACH(const ss_t& p, m_constants){
                log4cxx::MDC* mdc = new log4cxx::MDC(p.first, boost::lexical_cast<std::string>(p.second));
                v.push_back(mdc);
        }
        LOG4CXX_DEBUG(log, "monitor watchpoint");
        BOOST_FOREACH(log4cxx::MDC* mdc, v){
            delete mdc;
        }

    }
    void monitor::before_epoch(){
        if(m_every == 0 || !(m_cv_mode == CM_TRAIN || m_cv_mode == CM_TRAINALL))
            reset();
    }
    void monitor::register_gd(gradient_descent& gd){
        gd.after_epoch.connect( boost::bind(&monitor::after_epoch,this));
        gd.after_batch.connect( boost::bind(&monitor::after_batch,this, _1, _2));
        gd.before_epoch.connect( boost::bind(&monitor::before_epoch,this));

        BOOST_FOREACH(watchpoint* p, m_impl->m_watchpoints){
            switch(p->type){
                case WP_SINK:
                case WP_SCALAR_EPOCH_STATS:
                    gd.get_swiper().request_other_result(*(p->op), 0, false);
                default:
                    break;
            };
        }

        // the user probably registered variables with the monitor,
        // which attaches sinks. We need to recreate the swiper,
        // so that the sinks are updated accordingly.
        gd.repair_swiper(); 
    }

    void monitor::simple_logging()const{
        std::cout << "\r";
        if(m_cv_mode == CM_TRAIN)
            std::cout << "ANY";
        if(m_cv_mode == CM_TRAINALL)
            std::cout << "TRAINALL";
        if(m_cv_mode == CM_VALID)
            std::cout << "VAL";
        if(m_cv_mode == CM_TEST)
            std::cout << "TEST";
        int n_denominator = 0;
        BOOST_FOREACH(const watchpoint* p, m_impl->m_watchpoints){
            if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS || p->type == WP_FULL_WEIGHT_STATS  || p->type == WP_CONV_WEIGHT_STATS){
                n_denominator = boost::accumulators::count(p->stats(m_cv_mode));
                break;
            }
        }
        std::cout << " epoch "<<m_epochs<<":"<<m_batch_presentations<<"("<<n_denominator<<")"<<", ";
        //typedef std::pair<std::string, std::string> ss_t;
        //BOOST_FOREACH(const ss_t& p, m_constants){
        //    std::cout << p.first<<"="<<p.second<<" ";
        //}
        BOOST_FOREACH(const watchpoint* p, m_impl->m_watchpoints){
            if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS){
                std::cout  << p->name<<"="
                    << std::left << std::setprecision(4) << mean(p->name) <<", "//<<" ("
                    //<< std::left << std::setprecision(4) << stddev(p->name)<<"),  "
                    ;
            }
        }
        if(m_cv_mode == CM_VALID || m_cv_mode == CM_TEST)
            std::cout << "           " << std::endl;
        else
            std::cout << "           " << std::flush;
    }
}
