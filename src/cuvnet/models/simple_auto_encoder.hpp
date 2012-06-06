#ifndef __SIMPLE_AUTO_ENCODER_HPP__
#     define __SIMPLE_AUTO_ENCODER_HPP__

#include <boost/assign.hpp>
#include "generic_auto_encoder.hpp"

/**
 * This is a straight-forward symmetric auto-encoder
 */
class simple_auto_encoder 
: public generic_auto_encoder
{
    // these are the parametrs of the model
    boost::shared_ptr<Input>  m_weights, m_bias_h, m_bias_y;

    public:
    /// \f$ h := \sigma ( x W + b_h ) \f$
    virtual op_ptr  encode(op_ptr& inp){
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
     * @param initialize if true, the model will be initialized (false only for derived classes)
     */
    simple_auto_encoder(op_ptr& input, unsigned int hidden_dim, bool binary, bool initialize=true)
    :generic_auto_encoder(input,binary)
    {
        // ensure that we have shape information
        input->visit(determine_shapes_visitor()); 
        unsigned int input_dim = input->result()->shape[1];
        
        m_weights.reset(new Input(cuv::extents[input_dim][hidden_dim],"weights"));
        m_bias_h.reset(new Input(cuv::extents[hidden_dim],            "bias_h"));
        m_bias_y.reset(new Input(cuv::extents[input_dim],             "bias_y"));

        if(initialize) init();
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

/*
 *template<class T>
 *class simple_auto_encoder_weight_decay
 *: public T{
 *
 *    virtual op_ptr regularize(){ return sum(pow(m_weights,2.f)); }
 *    
 *};
 */


#endif /* __SIMPLE_AUTO_ENCODER_HPP__ */
