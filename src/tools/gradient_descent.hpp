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
    /**
     * does vanilla gradient descent: a loop over epochs and a weight update with a
     * learning rate/weight decay afterwards
     */
    struct gradient_descent{
        public:
            typedef std::vector<Op*> paramvec_t;
        protected:
            swiper           m_swipe;    ///< does fprop and bprop for us
            paramvec_t       m_params;   ///< all parameters w.r.t. which we optimize
            float            m_learnrate; ///< learnrate for weight updates
            float            m_weightdecay; ///< weight decay for weight updates
            unsigned int     m_rounds;    ///< number of rounds until optimum on validation set was attained
            std::map<Op*,cuv::tensor<float, cuv::host_memory_space> >    m_best_perf_params; ///< copies of parameters for current best performance
        public:
            /// triggered before an epoch starts. Should return number of batches!
            boost::signal<void(unsigned int)> before_epoch;
            /// triggered after an epoch finished
            boost::signal<void(unsigned int)> after_epoch;
            /// triggered before executing a batch (you should load batch data here!)
            boost::signal<void(unsigned int,unsigned int)> before_batch;
            /// triggered after executing a batch
            boost::signal<void(unsigned int,unsigned int)> after_batch;

            /// triggered when starting a validation epoch. Should return number of batches in validation set
            boost::signal<void(unsigned int)> before_validation_epoch;
            /// triggered after finishing a validation epoch
            boost::signal<void(unsigned int)> after_validation_epoch;

            /// should return current number of batches
            boost::signal<unsigned int(void)> current_batch_num;

            unsigned int rounds()const{ return m_rounds; }


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
                :m_swipe(*op, result, params), m_params(params), m_learnrate(learnrate), m_weightdecay(weightdecay)
                ,m_best_perf(std::numeric_limits<float>::infinity()), m_failed_improvement_rounds(0)
            { }

            /**
             * (virtual) destructor
             */
            virtual ~gradient_descent(){}

            boost::function<float(void)> m_performance;
            float                        m_best_perf;
            unsigned int                 m_failed_improvement_rounds;
            void early_stop_test(unsigned int every, float thresh, unsigned int maxfails, unsigned int current_epoch){
                if(current_epoch%every!=0)
                    return;
                validation_epoch(current_epoch);
                float perf = m_performance();
                std::cout << "   validation: "<< perf<<std::endl;
                if(perf < m_best_perf){
                    // save the (now best) parameters
                    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end(); it++){
                        Input* p = dynamic_cast<Input*>(*it);
                        cuvAssert(p);
                        m_best_perf_params[*it] = p->data();
                    }
                }
                if(perf <= m_best_perf-thresh){ // improve by at least thresh
                    m_best_perf = perf;
                    m_failed_improvement_rounds = 0;
                }else{
                    m_failed_improvement_rounds++;
                }
                if(m_failed_improvement_rounds>maxfails){
                    // load the best parameters again
                    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end(); it++){
                        Input* p = dynamic_cast<Input*>(*it);
                        cuvAssert(p);
                        std::map<Op*,cuv::tensor<float, cuv::host_memory_space> >::iterator mit = m_best_perf_params.find(*it);
                        if(mit != m_best_perf_params.end())
                             p->data() = m_best_perf_params[*it];
                    }

                    // save the number of rounds until minimum was attained
                    m_rounds = current_epoch - m_failed_improvement_rounds*every;
                    throw no_improvement_stop();
                }
                m_rounds = current_epoch;
            }
            /**
             * set up early stopping
             *
             * @param performance a function which determines how good we are after an epoch
             * @param performance
             */
            template<class T>
            void setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails){
                m_performance = performance;
                after_epoch.connect(boost::bind(&gradient_descent::early_stop_test,this,every_nth_epoch, thresh, maxfails, _1));
            }

            /**
             * this function should update all weights using backpropagated deltas
             */
            virtual void update_weights(){
                for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++){
                    cuvAssert(&((*it)->result()->delta.cdata()));
                    cuv::learn_step_weight_decay( ((Input*)*it)->data(), (*it)->result()->delta.cdata(), -m_learnrate, m_weightdecay);
                }
            }
            /**
             * runs a validation epoch
             */
            void validation_epoch(unsigned int current_epoch){
                before_validation_epoch(current_epoch);
                unsigned int n_batches = current_batch_num();
                for (unsigned int  batch = 0; batch < n_batches; ++batch) {
                    before_batch(current_epoch, batch);
                    m_swipe.fprop(); // fprop /only/
                    after_batch(current_epoch, batch);
                }
                after_validation_epoch(current_epoch);
            }
            /**
             * Does minibatch training.
             *
             * The signals \c before_epoch, \c after_epoch, 
             * \c before_batch, \c after_batch are executed as needed.
             *
             * @param n_epochs            how many epochs to run
             * @param n_batches_per_epoch how many batches there are in one epoch
             * @param update_every        after how many batches to update weights (set to 0 for `once per epoch'). Defaults to 1.
             * @param randomize           whether to randomize batches (default: true)
             */
            void minibatch_learning(const unsigned int n_epochs, unsigned int update_every=1, bool randomize=true){
                try{
                    std::vector<unsigned int> batchids;
                    for (unsigned int epoch = 0; epoch < n_epochs; ++epoch) {
                        unsigned int n_batches =  current_batch_num();
                        if(update_every==0)
                            update_every = n_batches;
                        if(n_batches!=batchids.size())
                            for(unsigned int i=0;i<n_batches;i++)
                                batchids.push_back(i);
                        if(randomize)
                            std::random_shuffle(batchids.begin(),batchids.end());
                        before_epoch(epoch);
                        for (unsigned int  batch = 0; batch < n_batches; ++batch) {
                            before_batch(epoch, batchids[batch]);
                            m_swipe.fprop();
                            m_swipe.bprop();
                            if((batch+1)%update_every == 0)
                                // TODO: accumulation does not work, currently delta is always overwritten!
                                update_weights(); 
                            after_batch(epoch, batchids[batch]);
                        }
                        after_epoch(epoch);
                    }
                }catch(no_improvement_stop){
                    ; // done.
                }
            }
            /**
             * Does batch training.
             *
             * The signals \c before_epoch, \c after_epoch, 
             * are executed as needed.
             *
             * @param n_epochs            how many epochs to run
             */
            void batch_learning(const unsigned int n_epochs){
                for (unsigned int epoch = 0; epoch < n_epochs; ++epoch) {
                    before_epoch(epoch);
                    m_swipe.fprop();
                    m_swipe.bprop();
                    update_weights();
                    after_epoch(epoch);
                }
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
