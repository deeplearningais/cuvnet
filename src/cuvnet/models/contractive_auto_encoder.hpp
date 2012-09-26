#ifndef __CONTRACTIVE_AUTO_ENCODER_HPP__
#     define __CONTRACTIVE_AUTO_ENCODER_HPP__

#include <boost/tuple/tuple.hpp>
#include "simple_auto_encoder.hpp"
#include "regularizers.hpp"

namespace cuvnet
{
/**
 * This is a contractive auto-encoder.
 *
 * @ingroup models
 */
class contractive_auto_encoder 
: public simple_auto_encoder
{
    /// how much regularization is required
    float m_reg;

    public:
    typedef simple_auto_encoder::op_ptr op_ptr;

    /**
     * constructor
     * 
     * @param hidden_dim the number of dimensions of the hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param reg regularization strength
     */
    contractive_auto_encoder(unsigned int hidden_dim, bool binary, float reg)
    :simple_auto_encoder(hidden_dim,binary)
    ,m_reg(reg)
    {
    }

    /**
     * Returns the L2 regularization loss.
     */
    virtual boost::tuple<float,op_ptr> regularize(){
        //op_ptr h_ = m_encoded*(1.f-m_encoded);
        op_ptr h_ = 1.f - pow(m_encoded, 2.f); // tanh
        return boost::make_tuple(
                m_reg,
                sum(  sum_to_vec(pow(h_,2.f),1)
                    * sum_to_vec(pow(m_weights,2.f),1))
                );
    }
};

/**
 * This is a contractive two-layer auto-encoder.
 *
 * @ingroup models
 */
class two_layer_contractive_auto_encoder 
: public two_layer_auto_encoder
{
    /// how much regularization is required
    float m_reg;

    /// stored so we can make it non-stochastic for testing
    boost::shared_ptr<RowSelector> m_rs;

    public:
    typedef two_layer_auto_encoder::op_ptr op_ptr;

    /**
     * turn stochastic row selection on/off.
     *
     * Mainly useful for derivative testing.
     */
    void set_stochastic(bool b){ m_rs->set_random(b); }

    /**
     * constructor
     * 
     * @param input the function that generates the input of the autoencoder
     * @param hidden_dim0 the number of dimensions of the 1st hidden layer
     * @param hidden_dim1 the number of dimensions of the 2nd hidden layer
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param reg regularization strength
     */
    two_layer_contractive_auto_encoder(unsigned int hidden_dim0, unsigned int hidden_dim1, bool binary, float reg)
    :two_layer_auto_encoder(hidden_dim0, hidden_dim1,binary)
    ,m_reg(reg)
    {
    }

    op_ptr m_schraudolph_reg;
    /**
     * Returns the L2 regularization loss.
     */
    virtual boost::tuple<float,op_ptr> regularize(){
        op_ptr contractive_loss;
        unsigned int bs = m_input->result()->shape[0];
        //unsigned int n_contrib = std::max(1u,bs/16);
        unsigned int n_contrib = 1;
        for(unsigned int i=0;i<n_contrib;i++){
            m_rs  = row_select(m_l0.m_y,m_l1.m_y); // select same (random) row in m_hl0 and m_hl1

            op_ptr J1 = m_l1.jacobian_x(result(m_rs,1));
            op_ptr J0 = m_l0.jacobian_x(result(m_rs,0));
            op_ptr tmp = mean(pow(prod(J1,J0,'t', 't'), 2.f));
            if(i == 0) 
                contractive_loss = tmp;
            else if(i == 1)
                contractive_loss = tmp + contractive_loss;
            else
                contractive_loss = add_to_param(contractive_loss,tmp);

        }
        m_schraudolph_reg = 
              m_l0.schraudolph_regularizer()
            + m_l1.schraudolph_regularizer();
        contractive_loss = contractive_loss + m_schraudolph_reg;
        //op_ptr J      = mat_times_vec(prod(mat_times_vec(m_weights1,h2_,1), m_weights2),h1_,0);
        //contractive_loss = sum( pow(J, 2.f) );

        return boost::make_tuple( 
                (m_reg * bs) / n_contrib, contractive_loss);
    }
};

}


#endif /* __CONTRACTIVE_AUTO_ENCODER_HPP__ */
