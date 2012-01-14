#ifndef __GRADIENT_DESCENT_HPP__
#     define __GRADIENT_DESCENT_HPP__

#include<boost/signals.hpp>
#include<cuvnet/op.hpp>
#include<cuvnet/op_utils.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>

namespace cuvnet
{

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
        public:
            /// triggered before an epoch starts
            boost::signal<void(unsigned int)> before_epoch;
            /// triggered after an epoch finished
            boost::signal<void(unsigned int)> after_epoch;
            /// triggered before executing a batch (you should load batch data here!)
            boost::signal<void(unsigned int,unsigned int)> before_batch;
            /// triggered after executing a batch
            boost::signal<void(unsigned int,unsigned int)> after_batch;

            /**
             * constructor
             * 
             * @param op     the function to be minimized
             * @param params the parameters w.r.t. which we should optimize
             * @param learnrate the learnrate for weight updates
             * @param weightdecay the weight decay for weight updates
             */
            gradient_descent(Op::op_ptr op, const paramvec_t& params, float learnrate=0.1f, float weightdecay=0.0f)
                :m_swipe(*op, true, params), m_params(params), m_learnrate(learnrate), m_weightdecay(weightdecay){ }

            /**
             * this function should update all weights using backpropagated deltas
             */
            virtual void update_weights(){
                for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++){
                    cuv::learn_step_weight_decay( ((Input*)*it)->data(), (*it)->result()->delta.cdata(), -m_learnrate, m_weightdecay);
                }
            }
            /**
             * Does minibatch training.
             *
             * The signals \c before_epoch, \c after_epoch, 
             * \c before_batch, \c after_batch are executed as needed.
             *
             * @param n_epochs            how many epochs to run
             * @param n_batches_per_epoch how many batches there are in one epoch
             */
            void minibatch_learning(const unsigned int n_epochs, const unsigned int n_batches_per_epoch){
                for (unsigned int epoch = 0; epoch < n_epochs; ++epoch) {
                    before_epoch(epoch);
                    for (unsigned int  batch = 0; batch < n_batches_per_epoch; ++batch) {
                        before_batch(epoch, batch);
                        m_swipe.fprop();
                        m_swipe.bprop();
                        update_weights();
                        after_batch(epoch, batch);
                    }
                    after_epoch(epoch);
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
             * @param params the parameters w.r.t. which we want to optimize op
             * @param learnrate the initial learningrate
             * @param weightdecay weight decay for weight updates
             */
        rprop_gradient_descent(Op::op_ptr op, const paramvec_t& params, float learnrate=0.0001f, float weightdecay=0.0f)
            :gradient_descent(op, params, learnrate, weightdecay), m_learnrates(params.size()), m_old_dw(params.size())
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
