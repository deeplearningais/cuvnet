#ifndef __DENOISING_AUTO_ENCODER_HPP__
#     define __DENOISING_AUTO_ENCODER_HPP__

#include "simple_auto_encoder.hpp"

/**
 * This is a straight-forward symmetric auto-encoder
 */
template<class Regularizer>
class denoising_auto_encoder 
: public simple_auto_encoder<Regularizer>
{
    /// how much noise to add to inputs
    float m_noise;

    public:
    typedef typename simple_auto_encoder<Regularizer>::op_ptr op_ptr;
    using   typename simple_auto_encoder<Regularizer>::m_binary;

    /**
     * @overload
     */
    virtual op_ptr  encode(op_ptr& inp){
        Op::op_ptr corrupt               = inp;
        if( m_binary && m_noise>0.f) corrupt =       zero_out(inp,m_noise);
        if(!m_binary && m_noise>0.f) corrupt = add_rnd_normal(inp,m_noise);
        return simple_auto_encoder<Regularizer>::encode(corrupt);
    }

    /**
     * constructor
     * 
     * @param input the function that generates the input of the autoencoder
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param noise if >0, add this much noise to input (type of noise depends on \c binary)
     * @param reg strength of regularization (usually some small positive constant)
     * @param initialize if true, the model will be initialized (false only for derived classes)
     */
    denoising_auto_encoder(op_ptr input, unsigned int hidden_dim, bool binary, float noise, float reg, bool initialize=true)
    :generic_auto_encoder(input,binary)
    ,simple_auto_encoder<Regularizer>(input,hidden_dim,binary,reg, false)
    ,m_noise(noise)
    {
        if(initialize) this->init(reg);
    }
};


#endif /* __DENOISING_AUTO_ENCODER_HPP__ */
