#ifndef __SIMPLE_AUTO_ENCODER_HPP__
#     define __SIMPLE_AUTO_ENCODER_HPP__

#include <boost/assign.hpp>
#include "generic_auto_encoder.hpp"

/**
 * This is a straight-forward symmetric auto-encoder
 */
template<class Regularizer>
class simple_auto_encoder 
: virtual public generic_auto_encoder
, public Regularizer
{
    // these are the parametrs of the model
    boost::shared_ptr<Input>  m_weights, m_bias_h, m_bias_y;

    unsigned int m_hidden_dim;

    public:
    /// \f$ h := \sigma ( x W + b_h ) \f$
    virtual op_ptr  encode(op_ptr& inp){
        if(!m_weights){
            inp->visit(determine_shapes_visitor()); 
            unsigned int input_dim = inp->result()->shape[1];
            
            m_weights.reset(new Input(cuv::extents[input_dim][m_hidden_dim],"weights"));
            m_bias_h.reset(new Input(cuv::extents[m_hidden_dim],            "bias_h"));
            m_bias_y.reset(new Input(cuv::extents[input_dim],             "bias_y"));
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

    /*
     * @return weight matrix
     */
    boost::shared_ptr<Input> get_weights(){return m_weights;} 

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
    ,Regularizer(binary)
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

/**
 * this just implements a weight decay to be mixed into a simple auto encoder
 */
class simple_auto_encoder_weight_decay
: virtual public generic_auto_encoder
{
    public:
        simple_auto_encoder_weight_decay(bool binary)
        : generic_auto_encoder(binary){}
    /**
     * L2 regularization of all two-dimensional parameter matrices 
     */
    virtual op_ptr regularize(){ 
        op_ptr regloss;
        BOOST_FOREACH(Op* op, unsupervised_params()){
            if(dynamic_cast<Input*>(op)->data().ndim()==2){
                op_ptr tmp = sum(pow(op->shared_from_this(),2.f)); 
                if(!regloss)
                    regloss = tmp;
                else
                    regloss = regloss + tmp;
            }
        }
        return regloss;
    }
};

/**
 * no regularization
 */
class simple_auto_encoder_no_regularization
: virtual public generic_auto_encoder
{
    public:
        simple_auto_encoder_no_regularization(bool binary)
        : generic_auto_encoder(binary){}
    /**
     * no regularization
     */
    virtual op_ptr regularize(){ 
        return op_ptr();
    }
};

#endif /* __SIMPLE_AUTO_ENCODER_HPP__ */
