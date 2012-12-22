#ifndef __GRADIENT_DESCENT_HPP__
#     define __GRADIENT_DESCENT_HPP__

#include<boost/signals.hpp>
#include<boost/bind.hpp>
#include<cuvnet/op.hpp>
#include<cuvnet/op_utils.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>

namespace cuvnet
{

    /**
     * @addtogroup learning_exceptions
     * @{
     */
    /// exception thrown when learning stopped.
    class gradient_descent_stop : public std::exception {};
    /// exception thrown when learning stopped due to NaNs.
    class arithmetic_error_stop : public gradient_descent_stop {};
    /// exception thrown when learning stopped in early_stopping.
    class no_improvement_stop : public gradient_descent_stop {};
    /// exception thrown when learning stopped due to convergence.
    class convergence_stop : public gradient_descent_stop {};
    /// exception thrown when learning stopped due to maximum iterations reached.
    class max_iter_stop : public gradient_descent_stop {};
    /// exception thrown when learning stopped due to time limit reached.
    class timeout_stop  : public gradient_descent_stop {};
    /// exception thrown when learning stopped due to a signal by a network peer.
    class network_stop  : public gradient_descent_stop {};
    /**
     * @}
     */


    /**
     * Does vanilla gradient descent: a loop over epochs and a weight update with a
     * learning rate/weight decay afterwards.
     * @ingroup gd
     */
    struct gradient_descent{
        public:
            typedef std::vector<Op*> paramvec_t;
        protected:
            Op::op_ptr       m_loss;     ///< the loss op we want to minimize
            unsigned int     m_result;   ///< the number of the result of the loss op we want to minimize
            paramvec_t       m_params;   ///< all parameters wrt which we optimize
            float            m_learnrate; ///< learnrate for weight updates
            //float            m_learnrate_decay; ///< factor by which lr is multiplied after each epoch
            float            m_weightdecay; ///< weight decay for weight updates
            unsigned long int  m_epoch;    ///< number of rounds until optimum on early-stopping set was attained
            std::map<Op*,cuv::tensor<float, cuv::host_memory_space> >    m_best_perf_params; ///< copies of parameters for current best performance
            unsigned int     m_epoch_of_saved_params; ///< stores the time of the saved params
            swiper           m_swipe;    ///< does fprop and bprop for us
        public:
            /// triggered before an epoch starts.
            boost::signal<void(unsigned int, unsigned int)> before_epoch;
            /// triggered after an epoch finished
            boost::signal<void(unsigned int, unsigned int)> after_epoch;
            /// triggered before executing a batch (you should load batch data here!)
            boost::signal<void(unsigned int,unsigned int)> before_batch;
            /// triggered after executing a batch
            boost::signal<void(unsigned int,unsigned int)> after_batch;
            /// triggered before updating weights
            boost::signal<void(unsigned int)> before_weight_update;
            /// triggered after updating weights
            boost::signal<void(unsigned int)> after_weight_update;

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
             * (virtual) destructor.
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
            void batch_learning(const unsigned int n_epochs, unsigned long int n_max_secs=INT_MAX);

            /**
             * return the epoch where \c best_perf was attained.
             */
            inline unsigned int epoch_of_saved_params(){
                return m_epoch_of_saved_params;
            }

            /**
             * decay learnrate by factor.
             */
            inline void decay_learnrate(float fact=0.98){
                m_learnrate *= fact;
            }

            /**
             * decrease learnrate by value.
             */
            inline void decrease_learnrate(float val){
                m_learnrate = std::max(1e-8f, m_learnrate-val);
            }

            /**
             * @return the current learnrate.
             */
            inline float learnrate()const{
                return m_learnrate;
            }


            /**
             * save the current parameters (on host) for retrieval
             * eg if the performance becomes worse.
             */
            void save_current_params();

            /**
             * runs an early-stopping epoch.
             *
             * @return number of early-stopping batches
             */
            void eval_epoch(unsigned int current_epoch);
            
        protected:
            /**
             * this function should update all weights using backpropagated deltas.
             */
            virtual void update_weights();

            /**
             * load the saved parameters back into the function.
             */
            void load_best_params();

    };

    /**
     * determine whether learning converged and throw an exception if it did.
     *
     * @ingroup gd_util
     */
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

            /// when failed this many steps, throw convergence_stop
            unsigned int m_max_steps;

            /// counts the number of convergences (up to m_max_steps)
            unsigned int m_steps;

            /// decrease learnrate by this much if convergence reached, but not m_max_steps
            float m_lr_fact;
            
            /// connection of convergence checker
            boost::signals::connection m_connection;


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
             * @param min_wups initial value for patience
             * @param patience_inc_fact patience is multiplied by this when better performance is found
             */
            convergence_checker(
                    gradient_descent& gd,
                    boost::function<float(void)> performance,
                    float thresh=0.95f, unsigned int min_wups=100, float patience_inc_fact=2.f);

            /**
             * modify learning rate upon convergence.
             *
             * @param max_steps converge this many times before stopping training
             * @param lr_fact multiply learning rate with this factor upon convergence
             */
            void decrease_lr(unsigned int max_steps=4, float lr_fact=0.75f);
            
            /**
             * test for convergence. 
             */
            void operator()(unsigned int current_epoch, unsigned int wups);
            
            /**
             * disconnect from gradient descent object we registered with.
             */
            void disconnect();
    };

    /**
     * The early_stopper stops gradient descent when the error on the
     * validation set does not decrease anymore.
     *
     * To use it, you have to create an early_stopper and let it live for as
     * long as your gradient descent lives. It automatically registers itself
     * with the supplied gradient descent object.
     *
     * @ingroup gd_util
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
            
            /// connection of early stopping
            boost::signals::connection m_connection;

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
            void operator()(unsigned int current_epoch, unsigned int wups);

            /**
             * return the best value we got during early_stopping.
             */
            inline float best_perf(){
                return m_best_perf;
            }

            /**
             * set the (initial?) patience (in weight updates).
             */
            inline void set_patience(unsigned int patience){
                m_patience = patience;
            }


            /**
             * disconnects early stopping from gradient_descent object we registered with. 
             */
            void disconnect();
    };

    /**
     * does rprop gradient descent
     *
     * also allocates and manages variables for learning rates
     * and old gradients for each parameter.
     *
     * @ingroup gd
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
         * updates the weights RPROP-style.
         */
        virtual void update_weights();
    };

    /**
     * does momentum gradient descent
     *
     * also allocates and manages variables for momentum
     *
     * @ingroup gd
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
         * updates the weights with momentum.
         */
        virtual void update_weights();

    };
    /**
     * does RMSPROP gradient descent.
     *
     * @ingroup gd
     */
    struct rmsprop_gradient_descent
    : public gradient_descent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            /// per-weight squared gradient sum
            std::vector<Op::value_type> m_sq_grad_sum; 
            /// numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
            float m_delta;
            /// gradient magnitude averaging constant (0.9 means mostly keep current average)
            float m_grad_avg_const;
            /// L1 penalty
            float m_l1penalty;
        public:
            /**
             * constructor
             *
             * @param op the function we want to minimize
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             * @param delta numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
             * @param grad_avg how much to stick to the old gradient magnitudes
             * @param l1penalty L1 penalty on parameters
             */
        rmsprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f, float delta=0.01f, float grad_avg=.9f, float l1_penalty=0.f);

        protected:
        /**
         * @overload
         * updates the weights with momentum.
         */
        virtual void update_weights();

    };
    /**
     * does AdaGrad gradient descent.
     *
     * @ingroup gd
     */
    struct adagrad_gradient_descent
    : public gradient_descent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            /// per-weight squared gradient sum
            std::vector<Op::value_type> m_sq_grad_sum; 
            /// numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
            float m_delta;
            /// after how many weight updates to reset squared gradient sums
            int m_winsize;
            /// how many weight updates have been performed so far
            int m_count;
            /// L1 penalty
            float m_l1penalty;
        public:
            /**
             * constructor
             *
             * @param op the function we want to minimize
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             * @param delta numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
             * @param winsize after how many weight updates to reset squared gradient sums
             * @param l1penalty L1 penalty on parameters
             */
        adagrad_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f, float delta=0.01f, int winsize=INT_MAX, float l1_penalty=0.f);

        protected:
        /**
         * @overload
         * updates the weights with momentum.
         */
        virtual void update_weights();

    };

    /**
     * Does Accelerated gradient descent.
     * This method supposedly works better for mini-batches. 
     * Introduced by Cotter et al., "Better Mini-Batch Algorithms via
     * Accelerated Gradient Methods" on NIPS.
     *
     * @ingroup gd
     */
    struct accelerated_gradient_descent
    : public gradient_descent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            /// this is the \f$w^{\mathrm{ag}}\f$ from the paper
            std::vector<Op::value_type> m_w_ag; 
            /// this is the \f$w\f$ from the paper, while the parameters used in the loss are \f$w^{md}\f$.
            std::vector<Op::value_type> m_w; 
            float m_beta;
            /// how many weight updates have been performed so far
            int m_count;
            /// L1 penalty
            float m_l1penalty;
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
        accelerated_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f);

        protected:
        /**
         * @overload
         * updates the weights with momentum.
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
     * @ingroup gd
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
#define UPDATE_ONLY_ON_SERVER 0
#if UPDATE_ONLY_ON_SERVER
                    matrix tmp = old_w[inp].copy();
#endif
                    cuv::apply_binary_functor(old_w[inp], inp->data(), cuv::BF_AXPBY, -1.f, 1.f);
                    
                    std::map<Op*, storage_t>::iterator upit = m_updates.find(inp);
                    if(upit != m_updates.end())
                        m_updates[inp] += (storage_t) old_w[inp];
                    else
                        m_updates[inp]  = (storage_t) old_w[inp];
#if UPDATE_ONLY_ON_SERVER
                    inp->data() = tmp;
#endif
                }
            }
    };
}

#endif /* __GRADIENT_DESCENT_HPP__ */