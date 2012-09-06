#ifndef __MONITOR_HPP__
#     define __MONITOR_HPP__

#include <datasets/dataset.hpp> // for cv_mode
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops.hpp>
#include <tools/function.hpp>
namespace cuvnet
{
    /// implementation of monitor
    struct monitor_impl;
    
    /// contains information tracked by monitor
    struct watchpoint;

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

            /// keeps all constants
            std::vector<std::pair<std::string,std::string> > m_constants; 

            /// counts the number of epochs we've seen
            unsigned int m_epochs;

            /// the training mode we're in currently
            cv_mode m_cv_mode;

            /// the split we're currently monitoring
            int m_split;

            /// file where we write the loss
            std::ofstream m_logfile;
            
            /// if true the header needs to be written in the file
            bool need_header_log_file;

            /// if true, log to stdout after each epoch
            bool m_verbose; 

            /// see pimpl-idiom
            boost::shared_ptr<monitor_impl> m_impl;

        public:
            /**
             * default ctor
             */
            monitor(bool verbose=false, const std::string& file_name = "loss.csv");

            /**
             * dtor destroys all watchpoints
             */
            ~monitor();

            /**
             * add a watch point
             *
             * @param type the type of the watchpoint, e.g. scalar stats or value sink
             * @param op   the op to watch
             * @param name a name by which the watchpoint can be identified
             */
            monitor& add(watchpoint_type type, boost::shared_ptr<Op> op, const std::string& name);

            template<class ValueType>
            monitor& add(const std::string& name, const ValueType& value){
                m_constants.push_back(std::make_pair(name, boost::lexical_cast<std::string>(value)));
                return *this;
            }


            /**
             * Sets the current mode of training.
             *
             * @warning the epochs counter is resetted here under two conditions:
             *    1) The split changes
             *    2) The mode changes to TRAINALL (since there might only be
             *       one split and condition 1 would not work)
             *
             * @param mode the mode we're in now
             * @param split the split we're working on now
             */
            void set_training_phase(cv_mode mode, int split=0);

            /**
             * increases number of batch presentations and updates scalar
             * statistics
             */
            void after_batch();

            /// resets all epoch statistics
            void before_epoch();

            /// increases number of epochs
            void after_epoch();

            /// @return the number of epochs this monitor has observed
            inline unsigned int epochs()             const{ return m_epochs;              }

            /// @return the number of batch presentations this monitor has observed
            inline unsigned int batch_presentations()const{ return m_batch_presentations; }

            /// get a watchpoint by name
            watchpoint& get(const std::string& name);

            /// get a const watchpoint by name
            const watchpoint& get(const std::string& name)const;
            
            /// return the number of examples of a named watchpoint 
            float count(const std::string& name)const;

            /// return the mean of a named watchpoint 
            float mean(const std::string& name)const;

            /// return the variance of a named watchpoint 
            float var(const std::string& name)const;

            /// return the standard deviation of a named watchpoint 
            float stddev(const std::string& name)const;

            /**
             * access a sink by a name
             * @return value of the first watchpoint with this name
             */
            const matrix& operator[](const std::string& name);

            /**
             * access a sink by a function pointer
             * @return value of the requested function
             */
            const matrix& operator[](const boost::shared_ptr<Op>& op);


            /**
             * plain text logging of all epochstats to the file 
             */
            void log_to_file();

            /**
             * plain text logging of all epochstats 
             */
            void simple_logging()const;
    };
}
#endif /* __MONITOR_HPP__ */
