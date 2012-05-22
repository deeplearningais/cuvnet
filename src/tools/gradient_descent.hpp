#ifndef __GRADIENT_DESCENT_HPP__
#     define __GRADIENT_DESCENT_HPP__

#include<boost/signals.hpp>
#include<boost/bind.hpp>
#include<boost/limits.hpp>
#include<cuvnet/op.hpp>
#include<cuvnet/op_utils.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>
#include<cuv/tensor_ops/rprop.hpp>

namespace cuvnet
{

    class no_improvement_stop : public std::exception {};
    class convergence_stop : public std::exception {};
    class max_iter_stop : public std::exception {};
    class timeout_stop  : public std::exception {};
    /**
     * does vanilla gradient descent: a loop over epochs and a weight update with a
     * learning rate/weight decay afterwards
     */
    struct gradient_descent{
        public:
            typedef std::vector<Op*> paramvec_t;
        protected:
            Op::op_ptr       m_loss;     ///< the loss op we want to minimize
            unsigned int     m_result;   ///< the number of the result of the loss op we want to minimize
            paramvec_t       m_params;   ///< all parameters w.r.t. which we optimize
            float            m_learnrate; ///< learnrate for weight updates
            float            m_learnrate_decay; ///< factor by which lr is multiplied after each epoch
            float            m_weightdecay; ///< weight decay for weight updates
            unsigned long int  m_epoch;    ///< number of rounds until optimum on early-stopping set was attained
            std::map<Op*,cuv::tensor<float, cuv::host_memory_space> >    m_best_perf_params; ///< copies of parameters for current best performance
            unsigned int     m_epoch_of_saved_params; ///< stores the time of the saved params
            swiper           m_swipe;    ///< does fprop and bprop for us
            bool             m_convergence_checking; ///< true if convergence checks are applied
            unsigned int     m_patience; ///< maximum number of epochs to run
        public:
            /// triggered before an epoch starts. Should return number of batches!
            boost::signal<void(unsigned int)> before_epoch;
            /// triggered after an epoch finished
            boost::signal<void(unsigned int)> after_epoch;
            /// triggered before executing a batch (you should load batch data here!)
            boost::signal<void(unsigned int,unsigned int)> before_batch;
            /// triggered after executing a batch
            boost::signal<void(unsigned int,unsigned int)> after_batch;

            /// triggered when starting a early-stopping epoch. Should return number of batches in early-stopping set
            boost::signal<void(unsigned int)> before_early_stopping_epoch;
            /// triggered after finishing a early-stopping epoch
            boost::signal<void(unsigned int)> after_early_stopping_epoch;

            /// should return current number of batches
            boost::signal<unsigned int(void)> current_batch_num;

            /// @return number of epochs we've run to obtain the minimum
            unsigned int iters()const{ return m_epoch; }

            /// repair the swiper, e.g. after using another swiper on the loss
            /// to dump parameters to a file
            void repair_swiper(){
                m_swipe = swiper(*m_loss, m_result, m_params);
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
            gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.1f, float weightdecay=0.0f)
                :m_loss(op), m_result(result), m_params(params), m_learnrate(learnrate), m_learnrate_decay(1.f), m_weightdecay(weightdecay)
                , m_epoch(0), m_epoch_of_saved_params(0)
                ,m_swipe(*op,result,params), m_convergence_checking(false)
                 , m_patience(4)
                ,m_best_perf(std::numeric_limits<float>::infinity())
            { 
            }

            /**
             * (virtual) destructor
             */
            virtual ~gradient_descent(){}

            /// should yield current performance
            boost::function<float(void)> m_performance; 
            /// smallest value of loss
            float                        m_best_perf;   

            /// this, multiplied by thresh parameter of early_stop_test 
            /// will give minimum improvement required for (not) early stopping
            float                        m_initial_performance; 

            /// last performance value (for convergence checking)
            float                        m_last_perf;

            /**
             * set up early stopping
             *
             * @param performance a function which determines how good we are after an epoch
             * @param every_nth_epoch run this check every n epochs
             * @param thresh stop when improvement is less than this much times initial value
             * @param patience_increase prolong training by this much if significant improvement found (e.g. 2 doubles training time)
             */
            template<class T>
            void setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, float patience_increase){
                m_performance = performance;
                before_epoch.connect(boost::bind(&gradient_descent::early_stop_test,this,every_nth_epoch, thresh, patience_increase, _1), boost::signals::at_front);
            }

            /**
             * set up stopping by convergence check
             *
             * @param performance a function which determines how good we are after an epoch
             * @param thresh stop when improvement is less than this much times initial value
             */
            template<class T>
            void setup_convergence_stopping(T performance, float thresh){
                m_performance = performance;
                m_convergence_checking = true;
                after_epoch.connect(boost::bind(&gradient_descent::convergence_test,this, thresh, _1), boost::signals::at_front);
            }


            /**
             * Does minibatch training.
             *
             * The signals \c before_epoch, \c after_epoch, 
             * \c before_batch, \c after_batch are executed as needed.
             *
             * @param n_iters             how many batch presentations
             * @param n_max_secs          maximum duration (seconds)
             * @param update_every        after how many batches to update weights (set to 0 for `once per epoch'). Defaults to 1.
             * @param randomize           whether to randomize batches (default: true)
             */
            void minibatch_learning(const unsigned int n_iters, unsigned long int n_max_secs=3600, unsigned int update_every=1, bool randomize=true){
                unsigned int n_batches = current_batch_num();
                if(update_every==0)
                    update_every = n_batches;
                std::vector<unsigned int> batchids;
                {   // prepare batch id vector
                    batchids.resize(n_batches);
                    for(unsigned int i=0;i<n_batches;i++)
                        batchids[i] = i;
                }
                unsigned long int iter = 0;
                try{
                    unsigned long int t_start = time(NULL);
                    for (m_epoch = 0; ; ++m_epoch) {
                        // stop if time limit is exceeded
                        if(time(NULL) - t_start > n_max_secs) {
                            std::cout << "Minibatch Learning Timeout ("<<(time(NULL)-t_start)<<"s)" << std::endl;/* cursor */
                            throw timeout_stop();
                        }
                        if(randomize)
                            std::random_shuffle(batchids.begin(),batchids.end());

                        before_epoch(m_epoch); // may run early stopping

                        for (unsigned int  batch = 0; batch < n_batches; ++batch) {

                            if(iter++ >= n_iters)
                                throw max_iter_stop();

                            before_batch(m_epoch, batchids[batch]); // should load data into inputs

                            m_swipe.fprop();  // forward pass

                            if(m_learnrate && !(m_convergence_checking && m_epoch==0)){
                                // this is not an evaluation pass, we're actually supposed to do work ;)
                                
                                m_swipe.bprop(); // backward pass

                                if((iter+1)%update_every == 0)
                                    // TODO: accumulation does not work, currently delta is always overwritten!
                                    update_weights(); 
                            }

                            after_batch(m_epoch, batchids[batch]); // should accumulate errors etc
                        }
                        after_epoch(m_epoch); // should log error etc

                        m_learnrate *= m_learnrate_decay;
                    }
                }catch(timeout_stop){
                }catch(no_improvement_stop){
                }catch(convergence_stop){
                }catch(max_iter_stop){
                }

                load_best_params();    // may also restore m_epoch
                m_epoch *= n_batches; // number of batch presentations
            }
            /**
             * Does batch training.
             *
             * The signals \c before_epoch, \c after_epoch, 
             * are executed as needed.
             *
             * @param n_epochs            how many epochs to run
             * @param n_max_secs          maximum duration (seconds)
             */
            void batch_learning(const unsigned int n_epochs, unsigned long int n_max_secs){
                unsigned long int t_start = time(NULL);
                for (unsigned int epoch = 0; epoch < n_epochs; ++epoch) {
                    if(time(NULL) - t_start > n_max_secs)
                        break;
                    before_epoch(epoch);
                    m_swipe.fprop();
                    m_swipe.bprop();
                    update_weights();
                    after_epoch(epoch);
                    m_learnrate *= m_learnrate_decay;
                }
            }

            /**
             * decay learnrate by factor
             */
            void decay_learnrate(float fact=0.98){
                m_learnrate_decay = fact;
            }
            
        private:
            /**
             * this function should update all weights using backpropagated deltas
             */
            virtual void update_weights(){
                for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++){
                    cuvAssert(&((*it)->result()->delta.cdata()));
                    Input* inp = (Input*) *it;

                    float lr = m_learnrate;
                    //if(inp->name().find("_wh")!=std::string::npos)
                        //lr /= 10.f;
                    //std::cout << inp->name()<<" delta: "<< cuv::norm2(inp->result()->delta.cdata())<< std::endl;[> cursor <]
                    //std::cout << inp->name()<<" value: "<< cuv::norm2(inp->data())<< std::endl;[> cursor <]
                    cuv::learn_step_weight_decay( inp->data(), inp->result()->delta.cdata(), -lr, m_weightdecay);
                }
            }
            /**
             * run an early-stopping test
             *
             * this function is called in the before_epoch hook.
             * Call @see setup_early_stopping to set it up.
             *
             * @param every   called every n-th epoch
             * @param thresh  determines "significant" performance improvements, i.e. 0.995
             * @param patience_increase prolong training by this much if significant improvement found (e.g. 2 doubles training time)
             * @param current_epoch number of current epoch
             */
            void early_stop_test(unsigned int every, float thresh, float patience_increase, unsigned int current_epoch){
                if(current_epoch%every!=0)
                    return;

                // this does the actual work, it runs fprop on all ES batches
                unsigned int n_batches = early_stopping_epoch(current_epoch);

                // determine how good we've been (usually set to some perf()
                // function of the construct you're trying to minimize)
                float perf = m_performance();

                if(current_epoch == 0)
                    m_initial_performance = perf;

                if(perf < m_best_perf)
                    std::cout << " * early-stopping(epoch "<<current_epoch<<" < "<<m_patience<<", "<<n_batches<<" batches): "<< perf<<std::endl;
                else
                    std::cout << " - early-stopping(epoch "<<current_epoch<<" < "<<m_patience<<", "<<n_batches<<" batches): "<< perf<<std::endl;

                if(perf < m_best_perf) {
                    // save the (now best) parameters
                    save_current_params();  
                    if(perf < thresh * m_best_perf){ 
                        // improved by more than thresh
                        m_patience = std::max((float)m_patience, (float)current_epoch * patience_increase);
                    }
                    m_best_perf = perf;
                }

                if(m_patience <= current_epoch){
                    // stop learning if we failed to get significant improvements
                    load_best_params();
                    throw no_improvement_stop();
                }
            }
            /**
             * save the current parameters (on host) for retrieval
             * e.g. if the performance becomes worse
             */
            void save_current_params(){
                    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end(); it++){
                        Input* p = dynamic_cast<Input*>(*it);
                        cuvAssert(p);
                        m_best_perf_params[*it] = p->data();
                    }
                    m_epoch_of_saved_params = m_epoch;
            }
            /**
             * load the saved parameters back into the function
             */
            void load_best_params(){
                    // load the best parameters again
                    bool did_load_something = false;
                    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end(); it++){
                        Input* p = dynamic_cast<Input*>(*it);
                        cuvAssert(p);
                        std::map<Op*,cuv::tensor<float, cuv::host_memory_space> >::iterator mit = m_best_perf_params.find(*it);
                        if(mit != m_best_perf_params.end()) {
                            std::cout << "...loading best `"<<p->name()<<"'"<<std::endl;
                            p->data() = m_best_perf_params[*it];
                            did_load_something = true;
                        }
                    }
                    if(did_load_something)
                        m_epoch = m_epoch_of_saved_params;
            }
            /**
             * test for convergence. 
             * use @see setup_convergence_stopping to use this function.
             */
            void convergence_test(float thresh, unsigned int current_epoch){
                float perf = m_performance();
                if(current_epoch == 0){
                    m_initial_performance = perf;
                    m_last_perf = perf;
                    return;
                }
                std::cout << "\r epoch: "<<current_epoch<<":  "<<perf<<" ("<<(m_last_perf-perf)/(thresh*m_initial_performance)<<")                  " << std::flush;
                if(m_last_perf - perf < thresh * m_initial_performance){
                    std::cout << std::endl << "Stopping due to convergence after "<<current_epoch<<" epochs, perf: "<< perf << std::endl;
                    throw convergence_stop();
                }
                m_last_perf = perf;
            }
            /**
             * runs an early-stopping epoch
             *
             * @return number of early-stopping batches
             */
            unsigned int early_stopping_epoch(unsigned int current_epoch){
                before_early_stopping_epoch(current_epoch);
                unsigned int n_batches = current_batch_num();
                for (unsigned int  batch = 0; batch < n_batches; ++batch) {
                    before_batch(current_epoch, batch);
                    m_swipe.fprop(); // fprop /only/
                    after_batch(current_epoch, batch);
                }
                after_early_stopping_epoch(current_epoch);
                return n_batches;
            }
    };
    /**
     * does rprop gradient descent
     *
     * also allocates and manages variables for learning rates
     * and old gradients for each parameter.
     */
    struct rprop_gradient_descent
    : public gradient_descent
    {
        private:
            std::vector<Op::value_type> m_learnrates; ///< per-weight learning rates
            std::vector<cuv::tensor<signed char,Op::value_type::memory_space_type> > m_old_dw;     ///< old delta-w signs
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
        rprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f)
            :gradient_descent(op, result, params, learnrate, weightdecay), m_learnrates(params.size()), m_old_dw(params.size())
        { 
            unsigned int i=0;
            for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
                m_learnrates[i].resize(((Input*)*it)->data().shape());
                m_learnrates[i] = learnrate;

                m_old_dw[i].resize(((Input*)*it)->data().shape());
                m_old_dw[i] = (signed char)0;
            }
        }
        /**
         * @overload
         * updates the weights RPROP-style
         */
        virtual void update_weights(){
            using namespace cuv;
            unsigned int i=0;
            for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
                Op::value_type dW = ::operator-((*it)->result()->delta.cdata()); // TODO: change sign in cuv::rprop
                cuv::rprop((dynamic_cast<Input*>(*it))->data(), dW, m_old_dw[i], m_learnrates[i], m_weightdecay, 0.0000000f);
            }
        }
    };
}

#endif /* __GRADIENT_DESCENT_HPP__ */
