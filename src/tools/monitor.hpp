#ifndef __MONITOR_HPP__
#     define __MONITOR_HPP__

#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuvnet/ops/output.hpp>

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
                WP_SCALAR_EPOCH_STATS  ///< keep stats over one epoch
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
                    :type(_t),op(_op),name(_name),sink(new Sink(name,op->result(result)))
                {
                }
                watchpoint_type         type;
                boost::shared_ptr<Op>     op;
                std::string             name;
                boost::shared_ptr<Sink> sink;  ///< this sink is managed by the monitor itself
                acc_t           scalar_stats;
            };

            boost::ptr_vector<watchpoint> m_watchpoints;

        public:
            /**
             * default ctor
             */
            monitor()
                :m_batch_presentations(0)
                ,m_epochs(0){ }


            /**
             * add a watch point
             *
             * @param type the type of the watchpoint, e.g. scalar stats or value sink
             * @param op   the op to watch
             * @param name a name by which the watchpoint can be identified
             */
            monitor& add(watchpoint_type type, boost::shared_ptr<Op>& op, const std::string& name){
                m_watchpoints.push_back(new watchpoint(type,op,name));
                return *this;
            }

            /**
             * increases number of batch presentations and updates scalar
             * statistics
             */
            void after_batch(){
                m_batch_presentations ++;

                BOOST_FOREACH(watchpoint& p, m_watchpoints){
                    if(p.type == WP_SCALAR_EPOCH_STATS)
                    {
                        p.scalar_stats((float)p.sink->cdata()[0]);
                        p.sink->forget();
                    }
                }
            }

            /// resets all epoch statistics
            void before_epoch(){
                BOOST_FOREACH(watchpoint& p, m_watchpoints){
                    if(p.type == WP_SCALAR_EPOCH_STATS)
                        p.scalar_stats = acc_t();
                }
                
            }

            /// increases number of epochs
            void after_epoch(){
                m_epochs ++;
            }

            /**
             * access a sink by a name
             * @return value of the first watchpoint with this name
             */
            const matrix& operator[](const std::string& name){
                BOOST_FOREACH(watchpoint& p, m_watchpoints){
                    if(p.name == name)
                        return p.sink->cdata();
                }
                throw std::runtime_error("Unknown watchpoint `"+name+"'");
            }

            /**
             * access a sink by a function pointer
             * @return value of the requested function
             */
            const matrix& operator[](const boost::shared_ptr<Op>& op){
                BOOST_FOREACH(watchpoint& p, m_watchpoints){
                    if(p.op == op)
                        return p.sink->cdata();
                }
                throw std::runtime_error("Unknown watchpoint requested");
            }
    };
}
#endif /* __MONITOR_HPP__ */
