#ifndef __GRADIENT_DESCENT_HPP__
#     define __GRADIENT_DESCENT_HPP__

#include<boost/signals2.hpp>
#include<boost/bind.hpp>
#include<cuvnet/op.hpp>
#include<cuvnet/op_utils.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>

namespace cuvnet
{
    class monitor;
    class early_stopper;

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
    /// exception thrown when the epoch ended (usually thrown in load_batch, ie in before_batch event)
    class epoch_end  : public gradient_descent_stop {};
    /**
     * @}
     */


    /**
     * Does vanilla gradient descent: a loop over epochs and a weight update with a
     * learning rate/weight decay afterwards.
     * @ingroup gd
     */
    class gradient_descent{
        public:
            typedef std::vector<Op*> paramvec_t;
            enum stop_reason_t {
                SR_NAN,
                SR_NO_IMPROVEMENT,
                SR_CONVERGENCE,
                SR_MAX_ITER,
                SR_TIMEOUT,
                SR_NETWORK,
                SR_EXTERNAL_REQUEST,
                SR_UNKNOWN,
            };
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
            unsigned int           m_update_every;    ///< determines weather to update after each batch if set to 1, and 0 otherwise
            stop_reason_t    m_stop_reason; ///< explains why learning was stopped
        public:
            /// triggered before an epoch starts.
            boost::signals2::signal<void(unsigned int, unsigned int)> before_epoch;
            /// triggered after an epoch finished
            boost::signals2::signal<void(unsigned int, unsigned int)> after_epoch;
            /// triggered before executing a batch (you should load batch data here!)
            boost::signals2::signal<void(unsigned int,unsigned int)> before_batch;
            /// triggered after executing a batch
            boost::signals2::signal<void(unsigned int,unsigned int)> after_batch;
            /// triggered before updating weights
            boost::signals2::signal<void(unsigned int)> before_weight_update;
            /// triggered after updating weights
            boost::signals2::signal<void(unsigned int)> after_weight_update;

            /// @return the reason why learning was stopped
            inline stop_reason_t stop_reason()const{return m_stop_reason;}

            /** request that training should stop asap.
             *  All events will still be executed, just the training epoch
             *  might become much shorter.
             *
             *  This is used in conjunction with a signal handler, e.g. for ctrl-c pressed.
             */
            inline void request_stop(){ m_stop_reason = SR_EXTERNAL_REQUEST; }

            /// @return the params we're optimizing
            inline const paramvec_t& params()const{return m_params;}

            /// set verbosity
            inline void set_verbosity(int verbosity){ 
                m_swipe.set_verbosity(verbosity);
            }

            /// triggered when finished with learning
            boost::signals2::signal<void()> done_learning;

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
            void minibatch_learning(const unsigned int n_max_epochs, unsigned long int n_max_secs=3600,  bool randomize=true);

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
             * set the current epoch.
             *
             * This is useful if we want to continue learning from an earlier
             * point in time. All functions which depend on the current epoch,
             * e.g. adapting the learning rates, can work correctly.
             */
            inline void set_epoch(unsigned int epoch){
                m_epoch = epoch;
            }

            /**
             * set the update every.
             *
             * Determines weather to update after each batch if set to 1, and 0 otherwise
             */
            inline void set_update_every(unsigned int update_every){
                m_update_every = update_every;
            }

            /**
             * get the update every.
             *
             * Determines weather to update after each batch if set to 1, and 0 otherwise
             */
            inline unsigned int get_update_every(){
                return m_update_every;
            }

            /**
             * return the epoch where \c best_perf was attained.
             */
            inline unsigned int epoch_of_saved_params(){
                return m_epoch_of_saved_params;
            }

            /**
             * @return the current learnrate.
             */
            inline float learnrate()const{
                return m_learnrate;
            }

            /**
             * set the learnrate to the given value
             * @param lr the new learnrate value
             */
            inline void set_learnrate(float lr){
                m_learnrate = lr;
            }


            /**
             * save the current parameters (on host) for retrieval
             * eg if the performance becomes worse.
             */
            void save_current_params();

            /**
             *  forget about "best" state saved by save_current_params().
             */
            void forget_best_params();

            /**
             * runs an early-stopping epoch.
             *
             * @return number of early-stopping batches
             */
            void eval_epoch(unsigned int current_epoch);

            /**
             * load the saved parameters back into the function.
             */
            void load_best_params();

            virtual void watch_evolution(boost::shared_ptr<cuvnet::monitor> mon, unsigned int vec_pos, unsigned int matrix_pos, std::string label);

        protected:
            /**
             * this function should update all weights using backpropagated deltas.
             */
            virtual void update_weights();
    };

    /**
     * determine whether learning converged and throw an exception if it did.
     *
     * @ingroup gd_util
     */
    class convergence_checker{
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
            boost::signals2::connection m_connection;


        public:

            /**
             * stopping by convergence on training set: ctor.
             *
             * Stops \c gradient_descent new value is more than
             * thresh*best_value for `min_wups` weight updates.
             *
             * `min_wups` is increased by a factor of
             * `patience_inc_fact` every time the threshold is
             * reached.
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
             * stopping by convergence on validation set: ctor.
             *
             * Stops \c gradient_descent new value is more than
             * thresh*best_value for `min_epochs` epochs.
             *
             * `min_epochs` is increased by a factor of
             * `patience_inc_fact` every time the threshold is
             * reached.
             *
             * Tested on /validation/ set, by referring to the perfomance measured in the early_stopper.
             * @param gd gradient_descent object to register with
             * @param es an early stopper to register with
             * @param performance a function which determines how good we are after an epoch
             * @param thresh stop when new value is more than thresh*best_value and no patience left
             * @param min_epochs initial value for patience
             * @param patience_inc_fact patience is multiplied by this when better performance is found
             */
            convergence_checker(
                    gradient_descent& gd,
                    early_stopper& es,
                    boost::function<float(void)> performance,
                    float thresh=0.95f, unsigned int min_epochs=100, float patience_inc_fact=2.f);

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
    class early_stopper{
        private:
            /// max num of epochs to run
            unsigned int  m_patience; 

            /// gradient descent object for calls to save_current_params
            /// and registering events
            gradient_descent& m_gd;

            /// should yield current performance
            boost::function<float(void)> m_performance; 

            /// a vector containing all validation set results together with the epoch they occurred in
            std::vector<std::pair<unsigned int, float> > m_val_perfs;

            /// best value attained so far
            float m_best_perf;

            /// the value that needs to be reached so that allowed time (=patience) is increased
            float m_perf_thresh;

            /// when to run.
            unsigned int m_every;

            /// when to stop.
            float m_thresh;

            /// how long to prolong training when below thresh
            float m_patience_increase;

            /// box filter for smoothing results
            unsigned int m_box_filter_size;
            
            /// connection of early stopping
            boost::signals2::connection m_connection;

            /// multiply learning rate for `m_max_steps` times before finally stopping learning
            unsigned int m_max_steps;

            /// multiply learning rate for `m_max_steps` by `m_lr_fact` before finally stopping learning
            float m_lr_fact;

        public:
            /// triggered when starting a early-stopping epoch.
            boost::signals2::signal<void(unsigned int)> before_early_stopping_epoch;

            /// triggered after finishing a early-stopping epoch
            boost::signals2::signal<void(unsigned int)> after_early_stopping_epoch;

            /// triggered when a new 'best' value is found
            boost::signals2::signal<void()> improved;

            /**
             * ctor.
             *
             * @param gd gradient_descent object to register with
             * @param performance a function which determines how good we are after an epoch,
             *        e.g. boost::bind(&monitor::mean, &mon, "classificatin loss")
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
            void operator()(unsigned int wups);

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
             * modify learning rate upon convergence.
             *
             * @param max_steps converge this many times before stopping training
             * @param lr_fact multiply learning rate with this factor upon convergence
             */
            inline void decrease_lr(unsigned int max_steps=4, float lr_fact=0.75f){
                m_max_steps = max_steps;
                m_lr_fact = lr_fact;
            }

            /**
             * @return the (estimated) performance value at a specific epoch.
             *
             * returns the value of the early-stopper run closest to the requested epoch.
             */
            float get_performance_at_epoch(unsigned int epoch);

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
            float m_l1decay;
            float m_eta_p;
            float m_eta_m;

        public:
            /**
             * constructor
             *
             * @param op the function we want to minimize
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             * @param l1decay sparsedecay parameter
             * @param eta_p increase-parameter for the learningrates
             * @param eta_m decrease-parameter for the learningrates
             */
        rprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f, float l1decay=0.0f, float eta_p=1.2f, float eta_m=0.5f);

        virtual void watch_evolution(boost::shared_ptr<cuvnet::monitor> mon, unsigned int vec_pos, unsigned int matrix_pos, std::string label);

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
    class momentum_gradient_descent
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

        inline float momentum(){ return m_momentum; }
        inline void set_momentum(float momentum){
        	m_momentum = momentum;
        	std::cout<<"momentum is"<<momentum<<std::endl;
        }
        void reset();

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
     * also allocates and manages variables for learning rates,
     * old gradients and squared gradient sum for each parameter.
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

        virtual void watch_evolution(boost::shared_ptr<cuvnet::monitor> mon, unsigned int vec_pos, unsigned int matrix_pos, std::string label);

        protected:
        /**
         * @overload
         * updates the weights with momentum.
         */
        virtual void update_weights();

    };
    /**
     * does Nesterov accelerated RMSPROP gradient descent.
     * (http://climin.readthedocs.org/en/latest/rmsprop.html)
     *
     * also allocates and manages variables for learning rates,
     * old gradients and squared gradient sum for each parameter.
     * 
     * @ingroup gd
     */
    struct na_rmsprop_gradient_descent
    : public gradient_descent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            /// per-weight uptdate of the previous learning-step
            std::vector<Op::value_type> m_oldW;
            /// per-weight squared gradient sum
            std::vector<Op::value_type> m_sq_grad_sum;
            ///< per-weight learning rates
            std::vector<Op::value_type> m_learnrates;
            /// momentum-constant
            float m_momentum;
            /// gradient magnitude averaging constant (0.9 means mostly keep current average)
            float m_grad_avg;
            ///adaptable step rate constant (multiply lr with (1+step_adapt) if update and momentum point into the same direction or (1-step_adapt) otherwise )
            float m_step_adapt;
            /// numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
            float m_delta;
            /// bounds for the learningrates
            float m_lr_max, m_lr_min;

        public:
            /**
             * constructor
             *
             * @param op the function we want to minimize
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             * @param momentum the momentum-constant
             * @param grad_avg how much to stick to the old gradient magnitudes
             * @param step_adapt adaptable step rate constant
             * @param delta numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
             * @param lr_max upper bound for learningrates
             * @param lr_min lower bound for learrningrates 
             */
        na_rmsprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay = 0.0f, float momentum=0.5f, float grad_avg = 0.5f, float step_adapt = 0.9f, float delta=0.01f, float lr_max = 0.1f, float lr_min = 0.00001f);

        virtual void watch_evolution(boost::shared_ptr<cuvnet::monitor> mon, unsigned int vec_pos, unsigned int matrix_pos, std::string label);

        protected:
        /**
         * @overload
         * updates the weights with momentum.
         */
        virtual void update_weights();

    };
    /**
     * does RRMSPROP gradient descent.
     * 
     * this rmsprop version is derived from rprop
     * 
     * also allocates and manages variables for learning rates,
     * old gradients and squared gradient sum for each parameter.
     * 
     * @ingroup gd
     * 
     */
    struct rrmsprop_gradient_descent
    : gradient_descent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
        private:
            /// per-weight squared gradient sum
            std::vector<Op::value_type> m_sq_grad_sum;
            /// per-weight learning rates
            std::vector<Op::value_type> m_learnrates;
            /// old delta-w signs
            std::vector<cuv::tensor<signed char,Op::value_type::memory_space_type> > m_old_dw;
            /// gradient magnitude averaging constant (0.9 means mostly keep current average)
            float m_grad_avg;
            /// numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
            float m_delta;
            ///increase- and decrease-parameter for the learningrates 
            float m_eta_p, m_eta_m;
            /// bounds for the learningrates
            float m_delta_max, m_delta_min;
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
             * @param l1decay L1 penalty on parameters
             * @param grad_avg how much to stick to the old gradient magnitudes
             * @param delta numerical stabilization constant: \f$H=\delta I+\|g\|_2\f$
             * @param eta_p increase-parameter for the learningrates
             * @param eta_m decrease-parameter for the learningrates
             * @param delta_max upper bound for learningrates
             * @param delta_min lower bound for learrningrates 
             */
        rrmsprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay = 0.0f, float l1decay=0.0f, float grad_avg = 0.5f, float delta=0.01f, float eta_p= 1.2f, float eta_m= 0.5f, float delta_max = 5.f, float delta_min = 0.00001f);

        virtual void watch_evolution(boost::shared_ptr<cuvnet::monitor> mon, unsigned int vec_pos, unsigned int matrix_pos, std::string label);

        protected:
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

        virtual void watch_evolution(boost::shared_ptr<cuvnet::monitor> mon, unsigned int vec_pos, unsigned int matrix_pos, std::string label);

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
            float m_beta;
            /// how many weight updates have been performed so far
            int m_count;
            /// step width = learnrate * iteration ^ m_p
            float m_p;
        public:
            /**
             * constructor
             *
             * @param op the function we want to minimize
             * @param result which result of op to minimize
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             * @param p the power to which iteration is raised (<1)
             */
        accelerated_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f, float p=0.5f);
        virtual void finish();

        protected:
        /**
         * @overload
         * updates the weights with momentum.
         */
        virtual void update_weights();

    };

    namespace detail
    {
        // this class is just to have member variables of
        // diff_recording_gradient_descent before the member variables of its
        // BaseGradientDescent, so we can use casts to
        // diff_recording_gradient_descent<gradient_descent>
        struct drgd_helper{
            protected:
                typedef cuv::host_memory_space storage_space;
                typedef cuv::tensor<float, storage_space> storage_t;
                std::map<Op*, storage_t> m_updates_;
                bool m_active_;
                monitor* m_mon_;
                drgd_helper():m_mon_(NULL){}
                virtual std::map<Op*, storage_t>& updates() {
                    return m_updates_;
                }
            public:
                virtual const bool& active() const{
                    return m_active_;
                }
                void set_active(bool b){ m_active_ = b; }
        };
        
        // this class is just to have member variables of
        // diff_recording_gradient_descent before the member variables of its
        // BaseGradientDescent, so we can use casts to
        // diff_recording_gradient_descent<gradient_descent>
        struct wrgd_helper{
            protected:
                typedef cuv::host_memory_space storage_space;
                typedef cuv::tensor<float, storage_space> storage_t;
                std::map<Op*, storage_t> m_updates_;
                std::map<Op*, float> m_vars_;
                std::map<Op*, float> m_avgnorm_;
                bool m_active_;
                monitor* m_mon_;
                unsigned int m_current_batch_;
                unsigned int m_every_;
                unsigned int m_n_rec_;
                wrgd_helper()
                    :m_mon_(NULL)
                    ,m_current_batch_(0)
                    ,m_every_(1)
                    ,m_n_rec_(0)
                {}
                virtual std::map<Op*, storage_t>& updates() {
                    return m_updates_;
                }
                virtual bool& active() {
                    return m_active_;
                }
                void set_active(bool b){ m_active_ = b; }
                void log(const std::string& desc, float val);
        };
    }

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
    :   public detail::drgd_helper,
        public BaseGradientDescent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
            using detail::drgd_helper::storage_space;
            using detail::drgd_helper::storage_t;
            using detail::drgd_helper::m_updates_;
            using detail::drgd_helper::m_active_;
        public:
            /** 
             * this template constructor provides perfect forwarding for all arguments.
             */
            template<typename... Params>
                diff_recording_gradient_descent(Params&&... args)
                : BaseGradientDescent(args...)
                {
                    m_active_ = true;
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
                this->after_batch.connect(boost::bind(t, &updates(), _1, _2));
            }

            template<class T>
            void set_sync_function_es(T t, early_stopper& es){
                this->after_batch.connect(boost::bind(t, &updates(), _1, _2));
                es.before_early_stopping_epoch.connect(boost::bind(
                            &detail::drgd_helper::set_active,
                            this, false));
                es.after_early_stopping_epoch.connect(boost::bind(
                            &detail::drgd_helper::set_active,
                            this, true));
            }



        protected:
            /**
             * A wrapper of BaseGradientDescent::update_weights that records
             * changes in the weights and accumulates them.
             *
             * @overload
             */
            virtual void update_weights(){

                if(m_active_)
                for(paramvec_t::iterator it=this->m_params.begin(); it!=this->m_params.end();it++){
                    ParameterInput* inp = (ParameterInput*) *it;
                    std::map<Op*, storage_t>::iterator upit = m_updates_.find(inp);
                    if(upit != m_updates_.end()){
                        cuv::apply_binary_functor(m_updates_[inp], (storage_t)inp->delta(), cuv::BF_XPBY, inp->get_learnrate_factor());
                    }else{
                        m_updates_[inp] = inp->delta();
                        m_updates_[inp] *= inp->get_learnrate_factor();
                    }
#define UPDATE_ONLY_ON_SERVER 0
#if UPDATE_ONLY_ON_SERVER
                    inp->delta() = 0.f;
#endif
                }
#if !UPDATE_ONLY_ON_SERVER
                BaseGradientDescent::update_weights();
#endif
            }
    };

    /** 
     * An aspect that allows to store the \b negative weight updates regardless of the underlying weight update mechanism.
     *
     * (Weight Update Recording Gradient Descent)
     *
     * Usage:
     * @code
     * wup_recording_gradient_descent<gradient_descent> gd(loss,0,params);
     * gd.minibatch_learning(...);
     * @endcode
     *
     * @ingroup gd
     */
    template<class BaseGradientDescent>
    struct wup_recording_gradient_descent
    :   public detail::wrgd_helper,
        public BaseGradientDescent
    {
        public:
            typedef std::vector<Op*> paramvec_t;
            using detail::wrgd_helper::storage_space;
            using detail::wrgd_helper::storage_t;
            using detail::wrgd_helper::m_updates_;
            using detail::wrgd_helper::m_active_;
            using detail::wrgd_helper::m_mon_;
            using detail::wrgd_helper::m_every_;
            using detail::wrgd_helper::m_n_rec_;
            using detail::wrgd_helper::m_avgnorm_;
            using detail::wrgd_helper::m_vars_;
            typedef wup_recording_gradient_descent<BaseGradientDescent> my_class;

        public:
            /** 
             * this template constructor provides perfect forwarding for all arguments.
             */
            template<typename... Params>
                wup_recording_gradient_descent(Params... args)
                : BaseGradientDescent(args...)
                {
                    m_active_ = true;
                    run_before_epoch();
                }

            void set_monitor(monitor& mon, unsigned int every = 1){
                m_mon_ = &mon;
                m_every_ = every;
            }



        protected:
            /**
             * A wrapper of BaseGradientDescent::update_weights that records
             * changes in the weights.
             *
             * @overload
             */
            virtual void update_weights(){
                if(m_active_ && (m_current_batch_ % m_every_ == 0) && m_mon_)
                for(paramvec_t::iterator it=this->m_params.begin(); it!=this->m_params.end();it++){
                    ParameterInput* inp = (ParameterInput*) *it;
                    if(cuv::IsSame<matrix::memory_space_type, detail::wrgd_helper::storage_space>::Result::value){
                        m_updates_[inp] = inp->cdata().copy();
                    }else{
                        m_updates_[inp] = inp->cdata();
                    }
                }

                // the standard weight update of the original gd
                BaseGradientDescent::update_weights();

                if(m_active_ && (m_current_batch_ % m_every_ == 0) && m_mon_)
                for(paramvec_t::iterator it=this->m_params.begin(); it!=this->m_params.end();it++){
                    ParameterInput* inp = (ParameterInput*) *it;
                    std::map<Op*, storage_t>::iterator upit = m_updates_.find(inp);
                    upit->second -= (host_matrix) inp->cdata();
                    
                    m_avgnorm_[inp] += cuv::norm1(upit->second) / upit->second.size();
                    m_vars_[inp] += cuv::var(upit->second);
                    m_n_rec_ ++;
                }

                if(m_active_ && (m_current_batch_ % m_every_ == 0) && m_mon_){
                    run_after_epoch();
                    run_before_epoch();
                }

                m_current_batch_ += 1;
            }

            void run_after_epoch(){
                if(m_active_ && m_mon_)
                for(paramvec_t::iterator it=this->m_params.begin(); it!=this->m_params.end();it++){
                    ParameterInput* inp = (ParameterInput*) *it;
                    //std::map<Op*, storage_t>::iterator upit = m_updates_.find(inp);
                    log(inp->name() + "_dvar", m_vars_[inp] / m_n_rec_);
                    log(inp->name() + "_davgnorm", m_avgnorm_[inp] / m_n_rec_);
                    log(inp->name() + "_drelavgnorm", m_avgnorm_[inp] / m_n_rec_ / (cuv::norm1(inp->cdata()) / inp->cdata().size()));
                }
            }

            void run_before_epoch(){
                m_n_rec_ = 0;
                for(paramvec_t::iterator it=this->m_params.begin(); it!=this->m_params.end();it++){
                    ParameterInput* inp = (ParameterInput*) *it;
                    m_avgnorm_[inp] = 0;
                    m_vars_[inp] = 0;
                }
            }
    };
}

#endif /* __GRADIENT_DESCENT_HPP__ */
