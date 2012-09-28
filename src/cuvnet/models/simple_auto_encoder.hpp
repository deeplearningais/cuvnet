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
        }

        if(!m_encoded)
            m_encoded =tanh(mat_plus_vec(
                        prod( inp, m_weights)
                        ,m_bias_h,1));
        return     m_encoded;
    }
    /// \f$  h W^T + b_y \f$
    virtual op_ptr  decode(op_ptr& enc){ 
        if(!m_decoded)
            m_decoded = mat_plus_vec(
                    prod( enc, m_weights, 'n','t')
                    ,m_bias_y,1);
        return m_decoded;
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
        //float diff = 4.f*std::sqrt(6.f/(input_dim+hidden_dim)); // logistic
        float diff = 2.f*std::sqrt(6.f/(input_dim+hidden_dim)); // tanh
        cuv::fill_rnd_uniform(m_weights->data());
        m_weights->data() *= 2*diff;
        m_weights->data() -=   diff;
        m_bias_h->data()   = 0.f;
        m_bias_y->data()   = 0.f;
    }
};

/**
 * A small module implementing the shortcut idea from Schraudolph98.
 *
 * \f$
 * y = (tanh(Bx) + \alpha Bx + \beta) + Cx
 * \f$
 */
struct shortcut_layer{

    shortcut_layer(unsigned int dim) :m_dim(dim){}
    unsigned int m_dim;

    typedef boost::shared_ptr<Op>     op_ptr;
    boost::shared_ptr<ParameterInput>  m_B; 
    boost::shared_ptr<ParameterInput>  m_C_enc; 
    boost::shared_ptr<ParameterInput>  m_C_dec; 
    boost::shared_ptr<ParameterInput>  m_alpha_enc;
    boost::shared_ptr<ParameterInput>  m_alpha_dec;
    boost::shared_ptr<ParameterInput>  m_beta_enc;
    boost::shared_ptr<ParameterInput>  m_beta_dec;
    boost::shared_ptr<ParameterInput>  m_bias_enc;
    boost::shared_ptr<ParameterInput>  m_bias_dec;

    op_ptr m_y, m_f_enc, m_f_dec;

    void reset_weights(){
        {
            unsigned int input_dim = m_B->data().shape(0);
            float diff = 2.f*std::sqrt(6.f/(input_dim+m_dim));   // tanh
            cuv::fill_rnd_uniform(m_B->data());
            m_B->data() *= 2*diff;
            m_B->data() -=   diff;
        }
        {
            /*
             *unsigned int input_dim = m_C_enc->data().shape(0);
             *float diff = 2.f*std::sqrt(6.f/(input_dim+m_dim));   // tanh
             *cuv::fill_rnd_uniform(m_C_enc->data());
             *m_C_enc->data() *= 2*diff;
             *m_C_enc->data() -=   diff;
             */
             m_C_enc->data() = 0.f;
             if(m_C_dec)
                 m_C_dec->data() = 0.f;
        }
        m_alpha_enc->data()  = -0.5f;
        m_beta_enc->data()   =  0.f;
        if(m_alpha_dec)
            m_alpha_dec->data()  = -0.5f;
        if(m_beta_dec)
            m_beta_dec->data()   =  0.f;
        m_bias_enc->data() =  0.f;
        m_bias_dec->data() =  0.f;
    }
    op_ptr m_Bx;
    op_ptr m_tanh_Bx;
    op_ptr m_tanh_BTx;
    op_ptr  encode(op_ptr inp){
        if(!m_B){
            inp->visit(determine_shapes_visitor()); 
            unsigned int input_dim = inp->result()->shape[1];

            m_B.reset(new ParameterInput(cuv::extents[input_dim][m_dim],"B"));
            m_C_enc.reset(new ParameterInput(cuv::extents[input_dim][m_dim],"C"));
            m_C_dec.reset(new ParameterInput(cuv::extents[m_dim][input_dim],"C'"));
            m_beta_enc.reset(new ParameterInput(cuv::extents[m_dim], "beta"));
            m_beta_dec.reset(new ParameterInput(cuv::extents[input_dim], "beta'"));
            m_alpha_enc.reset(new ParameterInput(cuv::extents[m_dim], "alpha"));
            m_alpha_dec.reset(new ParameterInput(cuv::extents[input_dim], "alpha'"));
            m_bias_enc.reset(new ParameterInput(cuv::extents[m_dim], "bias_enc"));
            m_bias_dec.reset(new ParameterInput(cuv::extents[input_dim], "bias_dec"));
        }

        m_Bx = mat_plus_vec(prod(inp, m_B), m_bias_enc, 1);
        m_tanh_Bx = tanh(m_Bx);
        op_ptr alphaBx = mat_times_vec(m_Bx, m_alpha_enc, 1);
        op_ptr alphaBx_plus_beta = mat_plus_vec(alphaBx, m_beta_enc, 1);
        op_ptr Cx = prod(inp, m_C_enc);
        m_f_enc = m_tanh_Bx + alphaBx_plus_beta;
        m_y = m_f_enc + Cx;


        return label("encoder_1l",m_y);
    }
    op_ptr jacobian_x(op_ptr enc){
        return mat_times_vec(m_B, 1.f-pow(enc, 2.f) + m_alpha_enc, 1) + m_C_enc;
    }
    op_ptr schraudolph_regularizer(){
        // f(x) and f'(x) should both be zero
        // derive w.r.t. alpha and beta
        op_ptr d_fenc_d_x = mat_plus_vec(1.f-pow(m_tanh_Bx,2.f), m_alpha_enc, 1);
        op_ptr res = label("schraudolph_1l_enc", mean(pow(m_f_enc, 2.f)) + mean(pow(d_fenc_d_x, 2.f)));
        if(m_tanh_BTx){
            op_ptr d_fdec_d_x = mat_plus_vec(1.f-pow(m_tanh_BTx,2.f), m_alpha_dec, 1);
            return res + label("schraudolph_1l_dec", mean(pow(m_f_dec, 2.f)) + mean(pow(d_fdec_d_x, 2.f)));
        }
        return res;
    }

    op_ptr decode(op_ptr enc, bool linear=false){
        if(linear){
            m_y = mat_plus_vec(prod(enc, m_B, 'n', 't'), m_bias_dec, 1);
            m_C_dec.reset();
            m_alpha_dec.reset();
            m_beta_dec.reset();
        }else{
            op_ptr res = mat_plus_vec(prod(enc, m_B, 'n', 't'), m_bias_dec, 1);
            m_tanh_BTx = tanh(res);
            m_f_dec = m_tanh_BTx + mat_plus_vec(mat_times_vec(res, m_alpha_dec, 1), m_beta_dec, 1);
            m_y = m_f_dec + prod(enc, m_C_dec);
        }
        return label("decoder_1l", m_y);
    }

    /**
     * Determine the parameters learned during pre-training
     * @overload
     */
    std::vector<Op*> unsupervised_params(){
        using namespace boost::assign;
        std::vector<Op*> v;
        v += m_B.get(), m_C_enc.get(), m_bias_dec.get(), m_bias_enc.get();
        v += m_alpha_enc.get(), m_beta_enc.get();
        if(m_C_dec){
            v += m_C_dec.get(), m_alpha_dec.get(), m_beta_dec.get();
        }
        return v;
    };

    /**
     * Determine the parameters learned during fine-tuning
     * @overload
     */
    std::vector<Op*> supervised_params(){
        using namespace boost::assign;
        std::vector<Op*> v;
        v += m_B.get(), m_C_enc.get(), m_bias_enc.get();
        v += m_alpha_enc.get(), m_beta_enc.get();
        return v;
    };


};

/**
 * This is a two-layer symmetric auto-encoder.
 *
 * @ingroup models
 */
class two_layer_auto_encoder 
: public generic_auto_encoder
{
    public:
        typedef boost::shared_ptr<Op>     op_ptr;

    public:
        // these are the parameters of the model
        shortcut_layer m_l0;
        shortcut_layer m_l1;

    public:

        /// \f$ h := \sigma ( x W + b_h ) \f$
        virtual op_ptr  encode(op_ptr& inp){
            return m_l1.encode(m_l0.encode(inp));
        }
        /// decode the encoded value given in `enc`
        virtual op_ptr  decode(op_ptr& enc){ 
            if(!m_decoded)
                m_decoded = m_l0.decode(m_l1.decode(enc), true);
            return m_decoded;
        }

        /**
         * Determine the parameters learned during pre-training
         * @overload
         */
        virtual std::vector<Op*> unsupervised_params(){
            std::vector<Op*> v0 = m_l0.unsupervised_params();
            std::vector<Op*> v1 = m_l1.unsupervised_params();
            v0.reserve(v0.size() + v1.size());
            v0.insert(v0.end(), v1.begin(), v1.end());
            return v0;
        };

        /**
         * Determine the parameters learned during fine-tuning
         * @overload
         */
        virtual std::vector<Op*> supervised_params(){
            std::vector<Op*> v0 = m_l0.supervised_params();
            std::vector<Op*> v1 = m_l1.supervised_params();
            v0.reserve(v0.size() + v1.size());
            v0.insert(v0.end(), v1.begin(), v1.end());
            return v0;
        };

        /**
         * constructor
         * 
         * @param input the function that generates the input of the autoencoder
         * @param hidden_dim0 the number of dimensions of the 1st hidden layer
         * @param hidden_dim1 the number of dimensions of the 2nd hidden layer
         * @param binary if true, assumes inputs are bernoulli distributed
         */
        two_layer_auto_encoder(unsigned int hidden_dim0, unsigned int hidden_dim1, bool binary)
            :generic_auto_encoder(binary)
             ,m_l0(hidden_dim0)
             ,m_l1(hidden_dim1)
    {
    }

        /**
         * initialize the weights with random numbers
         * @overload
         */
        virtual void reset_weights(){
            m_l0.reset_weights();
            m_l1.reset_weights();
        }
};

/**
 * L2-regularized simple auto-encoder.
 */
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
