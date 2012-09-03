#ifndef __AUTO_ENCODER_STACK_HPP__
#     define __AUTO_ENCODER_STACK_HPP__

#include "generic_auto_encoder.hpp"

namespace cuvnet
{
/**
 * A stack of auto-encoders.
 *
 * Useful for deep learning, where (classically) multiple auto-encoders are
 * chained, pre-trained and the top-level encoder is used as input for 
 * \c logistic_regression.
 *
 * @see generic_auto_encoder 
 * @see simple_auto_encoder
 * @see denoising_auto_encoder
 * @ingroup models
 */
class auto_encoder_stack
: public generic_auto_encoder
{
    public:
        typedef boost::shared_ptr<generic_auto_encoder > ae_type;
        typedef std::vector<ae_type> ae_vec_type;
        typedef boost::shared_ptr<Op>     op_ptr;

    private:
        ae_vec_type m_stack;
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                ar & boost::serialization::base_object<generic_auto_encoder>(*this);
                ar & m_stack;
            }
    public:

        /** ctor.
         * @param binary whether the inputs are Bernoulli-distributed
         */
        auto_encoder_stack(bool binary)
        :generic_auto_encoder(binary)
        {
        }

        /**
         * return the number of auto-encoders in the stack.
         */
        inline unsigned int size(){ return m_stack.size(); }

        /**
         * Determine the parameters learned during pre-training
         * @overload
         */
        virtual std::vector<Op*> unsupervised_params(){
            std::vector<Op*> v;
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it) {
                std::vector<Op*> tmp = (*it)->unsupervised_params();
                std::copy(tmp.begin(),tmp.end(),std::back_inserter(v));
            }
            return v;
        };

        /**
         * returns an op calculating the sum of regularizers for the stack for
         * use in supervised training.
         */
        op_ptr get_sum_of_regularizers(){
            int cnt=0;
            std::vector<op_ptr> ops;
            std::vector<float> args;
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it, ++cnt) {
                op_ptr r;
                float lambda;
                boost::tie(lambda, r)  = (*it)->regularize();
                if(!lambda || !r)
                    continue;
                ops.push_back(r);
                args.push_back(lambda);
            }

            // ugs. Try to limit the number of (x + a*y) double-ops using axpby.
            // An optimizer as in theano could do this automatically...
            op_ptr res;
            unsigned int size = 2 * (ops.size() / 2);
            for (unsigned int i = 0; i < size; i += 2){
                if(!res)
                    res =       axpby(args[i], ops[i], args[i+1], ops[i+1]);
                else
                    res = res + axpby(args[i], ops[i], args[i+1], ops[i+1]);
            }
            if(ops.size() != size)
                res = axpby(res, args.back(), ops.back());
            return res;
        }

        /**
         * Determine the parameters learned during fine-tuning
         * @overload
         */
        virtual std::vector<Op*> supervised_params(){
            std::vector<Op*> v;
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it) {
                std::vector<Op*> tmp = (*it)->supervised_params();
                std::copy(tmp.begin(),tmp.end(),std::back_inserter(v));
            }
            return v;
        };

        /**
         * initialize the weights with random numbers
         * @overload
         */
        virtual void reset_weights(){
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it) {
                (*it)->reset_weights();
            }
        }

        /**
         * @return the encoder of the last element in the stack
         * @overload
         */
        virtual op_ptr  encode(op_ptr& inp){
            assert(m_stack.size());
            op_ptr op;
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it) {
                bool is_first = it == m_stack.begin();

                if(is_first)
                    op = (*it)->encode(inp);
                else 
                    op = (*it)->encode(op);
            }
            return op;
        }

        /**
         * Decoder
         *
         * @overload
         * 
         * @param enc a function that is the encoder of the autoencoder
         * @return a function that decodes the encoded values
         */
        virtual op_ptr  decode(op_ptr& enc){ 
            assert(m_stack.size());
            op_ptr op;
            for (ae_vec_type::reverse_iterator  it = m_stack.rbegin(); it != m_stack.rend(); ++it) {
                bool is_first = it == m_stack.rend()-1;
                bool is_last  = it == m_stack.rbegin();

                if(0);
                else if(is_first && is_last)
                    op = (*it)->decode(enc);
                else if(is_first)
                    op = (*it)->decode(op);
                else if(is_last)
                    op = logistic((*it)->decode(enc));
                else 
                    op = logistic((*it)->decode(op));
            }
            return op;
        }

        /**
         * build auto-encoder stack by adding auto-encoders
         *
         * @param ae auto-encoder to be added to the stack
         */
        template<class T, typename... Params>
        auto_encoder_stack& add(Params... args){
            m_stack.push_back(boost::make_shared<T>(args...));
            return *this;
        }

        /**
         * construct encoder, decoder and loss; initialize weights.
         *
         * @overload
         */
        virtual void init(op_ptr input){
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it) {
                bool is_first = it == m_stack.begin();
                if(is_first)
                    (*it)->init(input);
                else
                    (*it)->init(boost::dynamic_pointer_cast<Op>((*(it-1))->get_encoded()));
            }
            generic_auto_encoder::init(input);
            reset_weights();
        }

        /**
         * deinit clears the losses (and thus the function graph),
         * but should not touch any parameters.
         *
         * @overload
         */
        virtual void deinit(){
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it) {
                (*it)->deinit();
            }
        }

        /**
         * return the n-th auto-encoder
         */
        inline generic_auto_encoder& get_ae(unsigned int i){ return *m_stack[i]; }

        /**
         * return the input to the stack
         */
        inline op_ptr input(){ return m_stack.front()->input(); }
};

} // namespace cuvnet


#endif /* __AUTO_ENCODER_STACK_HPP__ */
