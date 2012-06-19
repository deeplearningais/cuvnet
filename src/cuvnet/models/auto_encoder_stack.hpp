#ifndef __AUTO_ENCODER_STACK_HPP__
#     define __AUTO_ENCODER_STACK_HPP__

#include "generic_auto_encoder.hpp"

class simple_auto_encoder_no_regularization;
template<class Regularizer=simple_auto_encoder_no_regularization>

/**
 * A stack of auto-encoders.
 *
 * Useful for deep learning, where (classically) multiple auto-encoders are
 * chained, pre-trained and the top-level encoder is used as input for 
 * \c logistic_regression.
 *
 * @example logistic_regression.cpp
 *
 * @see generic_auto_encoder 
 * @see simple_auto_encoder
 * @see denoising_auto_encoder
 * @ingroup models
 */
class auto_encoder_stack
: virtual public generic_auto_encoder
, public Regularizer
{
    public:
        typedef boost::shared_ptr<generic_auto_encoder> ae_type;
        typedef std::vector<ae_type> ae_vec_type;

    private:
        ae_vec_type m_stack;
    public:

        auto_encoder_stack(bool binary)
        :generic_auto_encoder(binary)
        ,Regularizer(binary)
        {
        }

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
        virtual void init(op_ptr input, float regularization_strength=0.f){
            for (ae_vec_type::iterator  it = m_stack.begin(); it != m_stack.end(); ++it) {
                bool is_first = it == m_stack.begin();
                if(is_first)
                    (*it)->init(input, regularization_strength);
                else
                    (*it)->init(boost::dynamic_pointer_cast<Op>((*(it-1))->get_encoded()),regularization_strength);
            }
            generic_auto_encoder::init(input, regularization_strength);
            reset_weights();
        }
};


#endif /* __AUTO_ENCODER_STACK_HPP__ */
