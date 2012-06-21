#ifndef __DENOISING_AUTO_ENCODER_HPP__
#     define __DENOISING_AUTO_ENCODER_HPP__

#include "simple_auto_encoder.hpp"

/**
 * This is a straight-forward symmetric auto-encoder.
 *
 * @ingroup models
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
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param noise if >0, add this much noise to input (type of noise depends on \c binary)
     */
    denoising_auto_encoder(unsigned int hidden_dim, bool binary, float noise)
    :generic_auto_encoder(binary)
    ,simple_auto_encoder<Regularizer>(hidden_dim,binary)
    ,m_noise(noise)
    {
    }
};


#endif /* __DENOISING_AUTO_ENCODER_HPP__ */
