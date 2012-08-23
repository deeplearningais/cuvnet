#ifndef __DENOISING_AUTO_ENCODER_HPP__
#     define __DENOISING_AUTO_ENCODER_HPP__

#include <boost/tuple/tuple.hpp>
#include "simple_auto_encoder.hpp"
#include "regularizers.hpp"

namespace cuvnet
{
/**
 * This is a straight-forward symmetric auto-encoder.
 *
 * @ingroup models
 */
class denoising_auto_encoder 
: public simple_auto_encoder
{
    /// how much noise to add to inputs
    float m_noise;

    public:
    typedef simple_auto_encoder::op_ptr op_ptr;
    using   simple_auto_encoder::m_binary;

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
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param noise if >0, add this much noise to input (type of noise depends on \c binary)
     */
    denoising_auto_encoder(unsigned int hidden_dim, bool binary, float noise)
    :simple_auto_encoder(hidden_dim,binary)
    ,m_noise(noise)
    {
    }
};

struct l2reg_denoising_auto_encoder
: public denoising_auto_encoder
, public simple_weight_decay
{
    typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * constructor
     * 
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param noise if >0, add this much noise to input (type of noise depends on \c binary)
     * @param l2 regularization strength
     */
    l2reg_denoising_auto_encoder(unsigned int hidden_dim, bool binary, float noise, float l2)
    :denoising_auto_encoder(hidden_dim, binary, noise)
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
}


#endif /* __DENOISING_AUTO_ENCODER_HPP__ */
