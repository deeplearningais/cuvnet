#ifndef __MONITOR_HPP__
#     define __MONITOR_HPP__

#include <map>
#include <iomanip>
//#include <boost/ptr_container/ptr_vector.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops.hpp>
#include <cuv/tools/device_tools.hpp>
#include <tools/function.hpp>
#include <iostream>
#include <fstream>
namespace cuvnet
{
    /**
     * Monitors a function during learning, eg statistics over
     * certain function values like a loss. 
     *
     * It also manages the \c Sinks attached to a \c function.
     * This useful, if you want to dump or look at functions of intermediate
     * results.
     *
     * @ingroup tools
     */
    class monitor{
        typedef
            boost::accumulators::accumulator_set<double,
                boost::accumulators::stats<
                    boost::accumulators::tag::min,
                    boost::accumulators::tag::max,
                    boost::accumulators::tag::mean,
                    boost::accumulators::tag::variance(boost::accumulators::lazy) > > acc_t;
        public:
            /// The monitor supports different types of watchpoints:
            enum watchpoint_type {
                WP_SINK,                ///< simply create a sink which keeps the values 
                WP_SCALAR_EPOCH_STATS, ///< keep stats over one epoch
                WP_D_SINK,                ///< simply create a sink which keeps the values 
                WP_D_SCALAR_EPOCH_STATS, ///< keep stats over one epoch
                WP_FUNC_SINK,                ///< a sink which needs to be evaluated first
                WP_FUNC_SCALAR_EPOCH_STATS  ///< needs evaluation first, keeps stats over one epoch
            };
        private:
            /// counts the number of batches we've seen
            unsigned int m_batch_presentations;

            /// counts the number of epochs we've seen
            unsigned int m_epochs;

            /// if true we're in train phase, otherwise test phase
            bool m_is_train_phase;

            /// file where we write the loss
            std::ofstream m_logfile;
            
            /// if true the header needs to be written in the file
            bool need_header_log_file;

            /// a catch-all representation of a watch point
            struct watchpoint{
                /** ctor
                 * @param _t  the type of the watchpoint (e.g. scalar stats or just value sink)
                 * @param _op the op we want to gather information about
                 * @param _name a name for this watchpoint
                 * @param result the number of the result of _op
                 */
                watchpoint(watchpoint_type _t, boost::shared_ptr<Op> _op, const std::string& _name, unsigned int result=0)
                    :type(_t),op(_op),name(_name)
                {
                    switch(type){
                        case WP_SINK:
                        case WP_SCALAR_EPOCH_STATS:
                            sink.reset(new Sink(name,op->result(result)));
                            break;
                        case WP_D_SINK:
                        case WP_D_SCALAR_EPOCH_STATS:
                            dsink = delta_sink(name, op, result);
                            break;
                        case WP_FUNC_SINK:
                        case WP_FUNC_SCALAR_EPOCH_STATS:
                            func.reset(op,result,name);
                            break;
                    }
                }
                watchpoint_type         type;
                function                func;
                boost::shared_ptr<Op>     op;
                std::string             name;
                boost::shared_ptr<DeltaSink> dsink;  ///< this sink is managed by the monitor itself
                boost::shared_ptr<Sink> sink;  ///< this sink is managed by the monitor itself
                acc_t           scalar_stats;
            };

            std::vector<watchpoint*> m_watchpoints; ///< keeps all watchpoints 
            std::map<std::string, watchpoint*> m_wpmap;  ///< access watchpoints by name
            bool m_verbose; ///< if true, log to stdout after each epoch

        public:
            /**
             * default ctor
             */
            monitor(bool verbose=false, const std::string& file_name = "loss.csv")
                :m_batch_presentations(0)
                ,m_epochs(0)
                ,m_is_train_phase(true)
                ,m_logfile(file_name.c_str(), std::ios::out | std::ios::app)
                ,need_header_log_file(true)
                ,m_verbose(verbose){ }

            /**
             * dtor destroys all watchpoints
             */
            ~monitor(){
                BOOST_FOREACH(watchpoint* p, m_watchpoints){
                    delete p;
                }
            }

            /**
             * add a watch point
             *
             * @param type the type of the watchpoint, e.g. scalar stats or value sink
             * @param op   the op to watch
             * @param name a name by which the watchpoint can be identified
             */
            monitor& add(watchpoint_type type, boost::shared_ptr<Op> op, const std::string& name){
                m_watchpoints.push_back(new watchpoint(type,op,name));
                m_wpmap[name] = m_watchpoints.back();
                return *this;
            }


            /**
             * it is set to true if it is training phase, otherwise is test phase 
             *
             * @param is_train true if it is training phase, otherwise is test phase 
             */
            void set_training_phase(bool is_train){
                m_is_train_phase = is_train;
            }

            /**
             * increases number of batch presentations and updates scalar
             * statistics
             */
            void after_batch(){
                m_batch_presentations ++;

                BOOST_FOREACH(watchpoint* p, m_watchpoints){
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

            /// resets all epoch statistics
            void before_epoch(){
                BOOST_FOREACH(watchpoint* p, m_watchpoints){
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

            /// increases number of epochs
            void after_epoch(){
                if(m_is_train_phase)
                    m_epochs ++;
                if(m_verbose){
                    log_to_file();
                    simple_logging();
                }
                m_batch_presentations = 0;
            }

            /// @return the number of epochs this monitor has observed
            inline unsigned int epochs()             const{ return m_epochs;              }

            /// @return the number of batch presentations this monitor has observed
            inline unsigned int batch_presentations()const{ return m_batch_presentations; }

            /// get a watchpoint by name
            watchpoint& get(const std::string& name){
                std::map<std::string, watchpoint*>::iterator it 
                    = m_wpmap.find(name);
                if(it != m_wpmap.end())
                    return *it->second;
                throw std::runtime_error("Unknown watchpoint `"+name+"'");
            }
            /// get a const watchpoint by name
            const watchpoint& get(const std::string& name)const{
                std::map<std::string, watchpoint*>::const_iterator it 
                    = m_wpmap.find(name);
                if(it != m_wpmap.end())
                    return *it->second;
                throw std::runtime_error("Unknown watchpoint `"+name+"'");
            }
            
            /// return the number of examples of a named watchpoint 
            float count(const std::string& name)const{
                return boost::accumulators::count(get(name).scalar_stats);
            }

            /// return the mean of a named watchpoint 
            float mean(const std::string& name)const{
                return boost::accumulators::mean(get(name).scalar_stats);
            }

            /// return the variance of a named watchpoint 
            float var(const std::string& name)const{
                return boost::accumulators::variance(get(name).scalar_stats);
            }
            /// return the standard deviation of a named watchpoint 
            float stddev(const std::string& name)const{
                return std::sqrt(boost::accumulators::variance(get(name).scalar_stats));
            }

            /**
             * access a sink by a name
             * @return value of the first watchpoint with this name
             */
            const matrix& operator[](const std::string& name){
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

            /**
             * access a sink by a function pointer
             * @return value of the requested function
             */
            const matrix& operator[](const boost::shared_ptr<Op>& op){
                BOOST_FOREACH(watchpoint* p, m_watchpoints){
                    if(p->op == op)
                        return p->sink->cdata();
                }
                throw std::runtime_error("Unknown watchpoint requested");
            }


            /**
             * plain text logging of all epochstats to the file 
             */
            void log_to_file(){
                assert(m_logfile.is_open());
                if(need_header_log_file){
                    m_logfile << "is_train,epoch,mem";
                    need_header_log_file = false;
                    BOOST_FOREACH(const watchpoint* p, m_watchpoints){
                        m_logfile << "," << p->name;
                    }
                    m_logfile << std::endl;
                }
                    
                m_logfile << m_is_train_phase << "," << m_epochs << "," << cuv::getFreeDeviceMemory();
                BOOST_FOREACH(const watchpoint* p, m_watchpoints){
                    if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS){
                        // writes to the file the loss
                            m_logfile  << "," <<  mean(p->name)  << "," << stddev(p->name);
                    }
                }
                m_logfile << std::endl;
            }

            /**
             * plain text logging of all epochstats 
             */
            void simple_logging()const{
                std::cout << "\r epoch "<<m_epochs<<"/"<<m_batch_presentations<<":  free_mb="<<cuv::getFreeDeviceMemory()/1024/1024<<",  ";
                BOOST_FOREACH(const watchpoint* p, m_watchpoints){
                    if(p->type == WP_SCALAR_EPOCH_STATS || p->type == WP_FUNC_SCALAR_EPOCH_STATS || p->type == WP_D_SCALAR_EPOCH_STATS){
                        std::cout  << p->name<<"="
                        << std::left << std::setprecision(4) << mean(p->name) <<" ("
                        << std::left << std::setprecision(4) << stddev(p->name)<<"),  ";
                    }
                }
                std::cout << "           " << std::flush;
            }
    };
}
#endif /* __MONITOR_HPP__ */
