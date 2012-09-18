#ifndef __GRADIENT_DESCENT_HPP__
#     define __GRADIENT_DESCENT_HPP__

#include<boost/signals.hpp>
#include<boost/bind.hpp>
#include<cuvnet/op.hpp>
#include<cuvnet/op_utils.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>

namespace cuvnet
{

    class gradient_descent_stop : public std::exception {};
    class arithmetic_error_stop : public gradient_descent_stop {};
    class no_improvement_stop : public gradient_descent_stop {};
    class convergence_stop : public gradient_descent_stop {};
    class max_iter_stop : public gradient_descent_stop {};
    class timeout_stop  : public gradient_descent_stop {};
    class network_stop  : public gradient_descent_stop {};


    /**
     * Does vanilla gradient descent: a loop over epochs and a weight update with a
     * learning rate/weight decay afterwards.
     * @ingroup learning
     */
    struct gradient_descent{
        public:
            typedef std::vector<Op*> paramvec_t;
        protected:
            Op::op_ptr       m_loss;     ///< the loss op we want to minimize
            unsigned int     m_result;   ///< the number of the result of the loss op we want to minimize
            paramvec_t       m_params;   ///< all parameters wrt which we optimize
            float            m_learnrate; ///< learnrate for weight updates
            float            m_learnrate_decay; ///< factor by which lr is multiplied after each epoch
            float            m_weightdecay; ///< weight decay for weight updates
            unsigned long int  m_epoch;    ///< number of rounds until optimum on early-stopping set was attained
            std::map<Op*,cuv::tensor<float, cuv::host_memory_space> >    m_best_perf_params; ///< copies of parameters for current best performance
            unsigned int     m_epoch_of_saved_params; ///< stores the time of the saved params
            swiper           m_swipe;    ///< does fprop and bprop for us
        public:
            /// triggered before an epoch starts.
            boost::signal<void(unsigned int)> before_epoch;
            /// triggered after an epoch finished
            boost::signal<void(unsigned int)> after_epoch;
            /// triggered before executing a batch (you should load batch data here!)
            boost::signal<void(unsigned int,unsigned int)> before_batch;
            /// triggered after executing a batch
            boost::signal<void(unsigned int,unsigned int)> after_batch;

            /// triggered when finished with learning
            boost::signal<void()> done_learning;

            /// should return current number of batches
            boost::function<unsigned int(void)> current_batch_num;

            /// @return number of epochs we've run to obtain the minimum
            unsigned int iters()const{ return m_epoch; }

            /// repair the swiper, e.g. after using another swiper on the loss
            /// to dump parameters to a file
            void repair_swiper(){
                m_swipe.init();
            }

            /// @return the swiper object (e.g. for dumping function graph to file)
            inline swiper& get_swiper(){
                return m_swipe;
            }

            /// return the loss currently being optimized
            inline Op::op_ptr loss(){
                return m_loss;
            }

            /**
             * constructor
             * 
             * @param op     the function to be minimized
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we should optimize
             * @param learnrate the learnrate for weight updates
             * @param weightdecay the weight decay for weight updates
             */
            gradient_descent(const Op::op_ptr& op, unsigned int result, const paramvec_t& params, float learnrate=0.1f, float weightdecay=0.0f);

            /**
             * (virtual) destructor
             */
            virtual ~gradient_descent();


            /**
             * Does minibatch training.
             *
             * The signals \c before_epoch, \c after_epoch, 
             * \c before_batch, \c after_batch are executed as needed.
             *
             * @param n_max_epochs        maximum number of epochs
             * @param n_max_secs          maximum duration (seconds)
             * @param update_every        after how many batches to update weights (set to 0 for `once per epoch'). Defaults to 1.
             * @param randomize           whether to randomize batches (default: true)
             *
             * \callgraph
             */
            void minibatch_learning(const unsigned int n_max_epochs, unsigned long int n_max_secs=3600, unsigned int update_every=1, bool randomize=true);

            /**
             * Does batch training.
             *
             * The signals \c before_epoch, \c after_epoch, 
             * are executed as needed.
             *
             * @param n_epochs            how many epochs to run
             * @param n_max_secs          maximum duration (seconds)
             */
            void batch_learning(const unsigned int n_epochs, unsigned long int n_max_secs);

            /**
             * return the epoch where \c best_perf was attained.
             */
            inline unsigned int epoch_of_saved_params(){
                return m_epoch_of_saved_params;
            }

            /**
             * decay learnrate by factor
             */
            inline void decay_learnrate(float fact=0.98){
                m_learnrate_decay = fact;
            }


            /**
             * save the current parameters (on host) for retrieval
             * e.g. if the performance becomes worse
             */
            void save_current_params();

            /**
             * runs an early-stopping epoch
             *
             * @return number of early-stopping batches
             */
            void eval_epoch(unsigned int current_epoch);
            
        protected:
            /**
             * this function should update all weights using backpropagated deltas
             */
            virtual void update_weights();

            /**
             * load the saved parameters back into the function
             */
            void load_best_params();

    };

    struct convergence_checker{
        private:
            /// gradient descent object for calls to save_current_params
            /// and registering events
            gradient_descent& m_gd;

            /// last performance value 
            float m_last_perf;

            /// should yield current performance
            boost::function<float(void)> m_performance; 

            /// when to stop
            float m_thresh;

            ///< max num of epochs to run
            unsigned int  m_patience; 

            /// how much to prolong stopping
            float m_patience_inc_fact;


        public:

            /**
             * stopping by convergence: ctor.
             *
             * Stops \c gradient_descent new value is more than
             * thresh*best_value for `many` epochs.
             *
             * Tested on /training/ set, which should always improve, except
             * for too large/too small learning rates.
             *
             * @param gd gradient_descent object to register with
             * @param performance a function which determines how good we are after an epoch
             * @param thresh stop when new value is more than thresh*best_value and no patience left
             * @param min_epochs initial value for patience
             * @param patience_inc_fact patience is multiplied by this when better performance is found
             */
            convergence_checker(
                    gradient_descent& gd,
                    boost::function<float(void)> performance,
                    float thresh=0.95f, unsigned int min_epochs=5, float patience_inc_fact=2.f);
            
            /**
             * test for convergence. 
             */
            void operator()(unsigned int current_epoch);

    };

    /**
     * The early_stopper stops gradient descent when the error on the
     * validation set does not decrease anymore.
     *
     * To use it, you have to create an early_stopper and let it live for as
     * long as your gradient descent lives. It automatically registers itself
     * with the supplied gradient descent object.
     *
     */
    struct early_stopper{
        private:
            /// max num of epochs to run
            unsigned int  m_patience; 

            /// gradient descent object for calls to save_current_params
            /// and registering events
            gradient_descent& m_gd;

            /// should yield current performance
            boost::function<float(void)> m_performance; 

            /// a vector containing all validation set results for smoothing
            std::vector<float> m_val_perfs;

            /// best value attained so far
            float m_best_perf;

            /// when to run.
            unsigned int m_every;

            /// when to stop.
            float m_thresh;

            /// how long to prolong training when below thresh
            float m_patience_increase;

            /// box filter for smoothing results
            unsigned int m_box_filter_size;

        public:
            /// triggered when starting a early-stopping epoch.
            boost::signal<void(unsigned int)> before_early_stopping_epoch;

            /// triggered after finishing a early-stopping epoch
            boost::signal<void(unsigned int)> after_early_stopping_epoch;

            /**
             * ctor.
             *
             * @param gd gradient_descent object to register with
             * @param performance a function which determines how good we are after an epoch
             * @param thresh  determines "significant" performance improvements, i.e. 0.995
             * @param every   called every n-th epoch
             * @param patience_increase prolong training by this much if significant improvement found (e.g. 2 doubles training time)
             * @param box_filter_size size of window used to filter performance results (1 is equivalent to no filtering)
             */
            early_stopper(
                    gradient_descent& gd,
                    boost::function<float(void)> performance,
                    float thresh, unsigned int every, float patience_increase, unsigned int box_filter_size=1);
            /**
             * test for early stopping. 
             */
            void operator()(unsigned int current_epoch);

            /**
             * return the best value we got during early_stopping
             */
            inline float best_perf(){
                return m_best_perf;
            }
    };

    /**
     * does rprop gradient descent
     *
     * also allocates and manages variables for learning rates
     * and old gradients for each parameter.
     *
     * @ingroup learning
     */
    struct rprop_gradient_descent
    : public gradient_descent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            std::vector<Op::value_type> m_learnrates; ///< per-weight learning rates
            std::vector<cuv::tensor<signed char,Op::value_type::memory_space_type> > m_old_dw;     ///< old delta-w signs

            unsigned int m_n_batches;

            void inc_n_batches(){ m_n_batches ++; }
        public:
            /**
             * constructor
             *
             * @param op the function we want to minimize
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             */
        rprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f);

        protected:
        /**
         * @overload
         * updates the weights RPROP-style
         */
        virtual void update_weights();
    };

    /**
     * does momentum gradient descent
     *
     * also allocates and manages variables for momentum
     *
     * @ingroup learning
     */
    struct momentum_gradient_descent
    : public gradient_descent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            std::vector<Op::value_type> m_last_delta; ///< per-weight momentum
            float m_momentum; 
        public:
            /**
             * constructor
             *
             * @param op the function we want to minimize
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             */
        momentum_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f, float momentum=0.9f);

        protected:
        /**
         * @overload
         * updates the weights with momentum
         */
        virtual void update_weights();

    };

    /** 
     * An aspect that allows to store gradient updates for asynchronous gradient descent.
     *
     * Usage:
     * @code
     * diff_recording_gradient_descent<gradient_descent> gd(loss,0,params);
     * gd.minibatch_learning(...);
     * @endcode
     *
     * @ingroup learning
     */
    template<class BaseGradientDescent>
    struct diff_recording_gradient_descent
    : public BaseGradientDescent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            typedef cuv::host_memory_space storage_space;
            typedef cuv::tensor<float, storage_space> storage_t;
            std::map<Op*, storage_t> m_updates;
        public:

            /** 
             * this template constructor provides perfect forwarding for all arguments.
             */
            template<typename... Params>
                diff_recording_gradient_descent(Params... args)
                : BaseGradientDescent(args...)
                {
                }

            /**
             * use this to connect eg a param_synchronizer.
             *
             * Example:
             * @code
             * diff_recording_gradient_descent<gradient_descent> gd(loss,0,params);
             * network_communication::client clt("127.0.0.1","testnc");
             * param_synchronizer ps(clt, 10, 20, params);
             * gd.set_sync_function(boost::ref(ps));
             * gd.minibatch_learning(...);
             * @endcode
             */
            template<class T>
            void set_sync_function(T t){
                this->after_batch.connect(boost::bind(t, &m_updates, _1, _2));
            }

        protected:
            /**
             * A wrapper of BaseGradientDescent::update_weights that records
             * changes in the weights and accumulates them.
             *
             * @overload
             */
            virtual void update_weights(){
                std::map<ParameterInput*,matrix> old_w;
                for(paramvec_t::iterator it=this->m_params.begin(); it!=this->m_params.end();it++){
                    ParameterInput* inp = (ParameterInput*) *it;
                    old_w[inp] = inp->data().copy();
                }

                BaseGradientDescent::update_weights();

                for(paramvec_t::iterator it=this->m_params.begin(); it!=this->m_params.end();it++){
                    ParameterInput* inp = (ParameterInput*) *it;
                    //matrix tmp = old_w[inp].copy();
                    cuv::apply_binary_functor(old_w[inp], inp->data(), cuv::BF_AXPBY, -1.f, 1.f);
                    
                    std::map<Op*, storage_t>::iterator upit = m_updates.find(inp);
                    if(upit != m_updates.end())
                        m_updates[inp] += (storage_t) old_w[inp];
                    else
                        m_updates[inp]  = (storage_t) old_w[inp];
                    //inp->data() = tmp;
                }
            }
    };
}

#endif /* __GRADIENT_DESCENT_HPP__ */
