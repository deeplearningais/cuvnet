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
    float m_reg_strength;
    op_ptr m_reg;

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
    ,m_reg_strength(reg)
    {
    }

    /**
     * Returns the L2 regularization loss.
     */
    virtual boost::tuple<float,op_ptr> regularize(){
        //op_ptr h_ = m_encoded*(1.f-m_encoded);
        if(!m_reg) {
            op_ptr h_ = 1.f - pow(m_encoded, 2.f); // tanh
            m_reg = 
                sum(  sum_to_vec(pow(h_,2.f),1)
                    * sum_to_vec(pow(m_weights,2.f),1));
        }
        return boost::make_tuple(m_reg_strength, m_reg);
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
    float m_reg_strength;

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
    ,m_reg_strength(reg)
    {
    }

    op_ptr m_schraudolph_reg;
    /**
     * Returns the L2 regularization loss.
     */
    virtual boost::tuple<float,op_ptr> regularize(){
        float schraudolph_fact = 1E6;

        op_ptr s0 = m_l0->schraudolph_regularizer();
        op_ptr s0a = m_l0a->schraudolph_regularizer();
        op_ptr s1 = m_l1->schraudolph_regularizer();
        op_ptr s1a = m_l1a->schraudolph_regularizer();
        if(s0){
            // at least s0 should be regularized if it is possible at all
            if(m_mode == AEM_UNSUPERVISED)
                m_schraudolph_reg = s0 + s1 + s0a;
            else if(m_mode == AEM_SUPERVISED)
                m_schraudolph_reg = s0 + s0a;
        }
                    

        if(m_mode == AEM_UNSUPERVISED){
            op_ptr contractive_loss;
            unsigned int bs = m_input->result()->shape[0];
            //unsigned int n_contrib = std::max(1u,bs/16);
            unsigned int n_contrib = 1;
            for(unsigned int i=0;i<n_contrib;i++){
                m_rs  = row_select(m_l0->m_y,m_l1->m_y); // select same (random) row in m_hl0 and m_hl1

                op_ptr J1 = label("J1", m_l1->jacobian_x(result(m_rs,1)));
                op_ptr J0 = label("J0", m_l0->jacobian_x(result(m_rs,0)));
                op_ptr tmp = sum(pow(prod(J1,J0,'t', 't'), 2.f));
                if(i == 0) 
                    contractive_loss = tmp;
                else if(i == 1)
                    contractive_loss = tmp + contractive_loss;
                else
                    contractive_loss = add_to_param(contractive_loss,tmp);
            }
            if(m_schraudolph_reg)
                return boost::make_tuple( m_reg_strength, 
                        axpby(
                            bs / (float) n_contrib, contractive_loss, 
                            schraudolph_fact,       m_schraudolph_reg));
            else
                return boost::make_tuple( 
                        m_reg_strength * bs / (float) n_contrib, 
                        contractive_loss);
        }else /*if(m_mode == AEM_SUPERVISED)*/{
            //return boost::make_tuple( 0.0 , op_ptr());
            return boost::make_tuple( schraudolph_fact,
                    m_schraudolph_reg);
        }
    }
};

}


#endif /* __CONTRACTIVE_AUTO_ENCODER_HPP__ */
