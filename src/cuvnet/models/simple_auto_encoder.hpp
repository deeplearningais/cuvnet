#ifndef __SIMPLE_AUTO_ENCODER_HPP__
#     define __SIMPLE_AUTO_ENCODER_HPP__

#include <boost/assign.hpp>
#include "regularizers.hpp"
#include <cuvnet/op_utils.hpp>
#include "generic_auto_encoder.hpp"

namespace cuvnet
{



/**
 * This is a straight-forward symmetric auto-encoder.
 *
 * @example auto_enc.cpp
 * @ingroup models
 */
class simple_auto_encoder 
: public generic_auto_encoder
{
    protected:
    // these are the parameters of the model
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
    :generic_auto_encoder(binary)
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

struct l2reg_simple_auto_encoder
: public simple_auto_encoder
, public simple_weight_decay
{
    typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * constructor
     * 
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param l2 regularization strength
     */
    l2reg_simple_auto_encoder(unsigned int hidden_dim, bool binary, float l2)
    :simple_auto_encoder(hidden_dim, binary)
    ,simple_weight_decay(l2)
    {
    }
    /**
     * Returns the L2 regularization loss.
     */
    virtual boost::tuple<float,op_ptr> regularize(){
        return boost::make_tuple(
                simple_weight_decay::strength(),
                simple_weight_decay::regularize(unsupervised_params()));
    }
};

} // namespace cuvnet
#endif /* __SIMPLE_AUTO_ENCODER_HPP__ */
