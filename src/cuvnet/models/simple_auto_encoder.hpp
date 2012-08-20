#ifndef __SIMPLE_AUTO_ENCODER_HPP__
#     define __SIMPLE_AUTO_ENCODER_HPP__

#include <boost/assign.hpp>
#include "generic_auto_encoder.hpp"

namespace cuvnet
{

/**
 * Implements a weight decay to be mixed into a simple auto encoder (etc)
 *
 * @ingroup models
 */
class simple_weight_decay
{
    public:
        simple_weight_decay(){}
        typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * L2 regularization of all two-dimensional parameter matrices 
     */
    virtual op_ptr regularize(const std::vector<Op*>& unsupervised_params){ 
        op_ptr regloss;
        int cnt = 0;
        BOOST_FOREACH(Op* op, unsupervised_params){
            if(dynamic_cast<ParameterInput*>(op)->data().ndim()==2){
                op_ptr tmp = mean(pow(op->shared_from_this(),2.f)); 
                if(cnt == 0)
                    regloss = tmp;
                else if(cnt == 1)
                    regloss = regloss + tmp;
                else
                    regloss = add_to_param(regloss, tmp);
                cnt ++ ;
            }
        }
        return regloss;
    }
};

/**
 * No regularization (mixin for auto-encoders etc).
 *
 * @ingroup models
 */
class no_regularization
{
    public:
        no_regularization(){}
        typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * no regularization
     */
    virtual op_ptr regularize(const std::vector<Op*>& unsupervised_params){ 
        return op_ptr();
    }
};


/**
 * This is a straight-forward symmetric auto-encoder.
 *
 * @example auto_enc.cpp
 * @ingroup models
 */
template<class Base=no_regularization>
class simple_auto_encoder 
: public generic_auto_encoder<Base>
{
    // these are the parametrs of the model
    boost::shared_ptr<ParameterInput>  m_weights, m_bias_h, m_bias_y;

    unsigned int m_hidden_dim;

    public:
    typedef boost::shared_ptr<Op>     op_ptr;
    /// \f$ h := \sigma ( x W + b_h ) \f$
    virtual op_ptr  encode(op_ptr& inp){
        if(!m_weights){
            inp->visit(determine_shapes_visitor()); 
            unsigned int input_dim = inp->result()->shape[1];
            
            m_weights.reset(new ParameterInput(cuv::extents[input_dim][m_hidden_dim],"weights"));
            m_bias_h.reset(new ParameterInput(cuv::extents[m_hidden_dim],            "bias_h"));
            m_bias_y.reset(new ParameterInput(cuv::extents[input_dim],             "bias_y"));
        }else{
            inp->visit(determine_shapes_visitor()); 
            unsigned int input_dim = inp->result()->shape[1];
            cuvAssert(m_weights->data().shape(0)==input_dim);
        }

        return logistic(mat_plus_vec(
                    prod( inp, m_weights)
                    ,m_bias_h,1));
    }
    /// \f$  h W^T + b_y \f$
    virtual op_ptr  decode(op_ptr& enc){ 
        return mat_plus_vec(
                prod( enc, m_weights, 'n','t')
                ,m_bias_y,1);
    }

    /**
     * @return weight matrix
     */
    boost::shared_ptr<ParameterInput> get_weights(){return m_weights;} 

    /**
     * Determine the parameters learned during pre-training
     * @overload
     */
    virtual std::vector<Op*> unsupervised_params(){
        return boost::assign::list_of(m_weights.get())(m_bias_h.get())(m_bias_y.get());
    };

    /**
     * Determine the parameters learned during fine-tuning
     * @overload
     */
    virtual std::vector<Op*> supervised_params(){
        return boost::assign::list_of(m_weights.get())(m_bias_h.get());
    };

    /**
     * constructor
     * 
     * @param input the function that generates the input of the autoencoder
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     */
    simple_auto_encoder(unsigned int hidden_dim, bool binary)
    :generic_auto_encoder<Base>(binary)
    ,m_hidden_dim(hidden_dim)
    {
    }

    /**
     * initialize the weights with random numbers
     * @overload
     */
    virtual void reset_weights(){
        unsigned int input_dim = m_weights->data().shape(0);
        unsigned int hidden_dim = m_weights->data().shape(1);
        float diff = 4.f*std::sqrt(6.f/(input_dim+hidden_dim));
        cuv::fill_rnd_uniform(m_weights->data());
        m_weights->data() *= 2*diff;
        m_weights->data() -=   diff;
        m_bias_h->data()   = 0.f;
        m_bias_y->data()   = 0.f;
    }
};


} // namespace cuvnet
#endif /* __SIMPLE_AUTO_ENCODER_HPP__ */
