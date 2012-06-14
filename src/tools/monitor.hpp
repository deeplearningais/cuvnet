#ifndef __MONITOR_HPP__
#     define __MONITOR_HPP__

#include <map>
//#include <boost/ptr_container/ptr_vector.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuvnet/ops/output.hpp>
#include <tools/function.hpp>

namespace cuvnet
{
    /**
     * This class monitors a function during learning, e.g. statistics over
     * certain function values like a loss. It also manages the sinks attached
     * to a function.
     */
    class monitor{
        typedef
            boost::accumulators::accumulator_set<double,
                boost::accumulators::stats<
                    boost::accumulators::tag::mean,
                    boost::accumulators::tag::variance(boost::accumulators::lazy) > > acc_t;
        public:
            /// The monitor supports different types of watchpoints:
            enum watchpoint_type {
                WP_SINK,                ///< simply create a sink which keeps the values 
                WP_SCALAR_EPOCH_STATS, ///< keep stats over one epoch
                WP_FUNC_SINK,                ///< a sink which needs to be evaluated first
                WP_FUNC_SCALAR_EPOCH_STATS  ///< needs evaluation first, keeps stats over one epoch
            };
        private:
            /// counts the number of batches we've seen
            unsigned int m_batch_presentations;

            /// counts the number of epochs we've seen
            unsigned int m_epochs;

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
            monitor(bool verbose=false)
                :m_batch_presentations(0)
                ,m_epochs(0)
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
             * increases number of batch presentations and updates scalar
             * statistics
             */
            void after_batch(){
                m_batch_presentations ++;

                BOOST_FOREACH(watchpoint* p, m_watchpoints){
                    if(p->type == WP_SCALAR_EPOCH_STATS)
                    {
                        p->scalar_stats((float)p->sink->cdata()[0]);
                        p->sink->forget();
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
                    if(p->type == WP_SCALAR_EPOCH_STATS)
                        p->scalar_stats = acc_t();
                    if(p->type == WP_FUNC_SCALAR_EPOCH_STATS)
                        p->scalar_stats = acc_t();
                }
                
            }

            /// increases number of epochs
            void after_epoch(){
                m_epochs ++;
                if(m_verbose)
                    simple_logging();
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

            /// return the mean of a named watchpoint 
            float mean(const std::string& name)const{
                return boost::accumulators::mean(get(name).scalar_stats);
            }

            /// return the variance of a named watchpoint 
            float var(const std::string& name)const{
                return boost::accumulators::variance(get(name).scalar_stats);
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
                    case WP_FUNC_SINK:
                    case WP_FUNC_SCALAR_EPOCH_STATS:
                        return wp.func.evaluate();
                }
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
             * plain text logging of all epochstats 
             */
            void simple_logging()const{
                std::cout << "\r epoch "<<m_epochs<<": ";
                BOOST_FOREACH(const watchpoint* p, m_watchpoints){
                    if(p->type == WP_SCALAR_EPOCH_STATS){
                        std::cout << p->name<<"="<<mean(p->name)<<", ";
                    }
                    std::cout << "           ";
                }
            }
    };
}
#endif /* __MONITOR_HPP__ */
