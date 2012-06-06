#ifndef __DENOISING_AUTO_ENCODER_HPP__
#     define __DENOISING_AUTO_ENCODER_HPP__

#include "simple_auto_encoder.hpp"

/**
 * This is a straight-forward symmetric auto-encoder
 */
struct denoising_auto_encoder 
: public simple_auto_encoder
{
    /// how much noise to add to inputs
    float m_noise;

    /**
     * @overload
     */
    virtual op_ptr  encode(op_ptr& inp){
        Op::op_ptr corrupt               = inp;
        if( m_binary && m_noise>0.f) corrupt =       zero_out(inp,m_noise);
        if(!m_binary && m_noise>0.f) corrupt = add_rnd_normal(inp,m_noise);
        return simple_auto_encoder::encode(corrupt);
    }

    /**
     * constructor
     * 
     * @param input the function that generates the input of the autoencoder
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param noise if >0, add this much noise to input (type of noise depends on \c binary)
     * @param initialize if true, the model will be initialized (false only for derived classes)
     */
    denoising_auto_encoder(op_ptr input, unsigned int hidden_dim, bool binary, float noise, bool initialize=true)
    :simple_auto_encoder(input,hidden_dim,binary,false)
    ,m_noise(noise)
    {
        if(initialize) init();
    }
};


#endif /* __DENOISING_AUTO_ENCODER_HPP__ */
