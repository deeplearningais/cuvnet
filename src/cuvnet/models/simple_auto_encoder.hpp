#ifndef __SIMPLE_AUTO_ENCODER_HPP__
#     define __SIMPLE_AUTO_ENCODER_HPP__

#include <boost/assign.hpp>
#include "regularizers.hpp"
#include <cuvnet/op_utils.hpp>
#include "generic_auto_encoder.hpp"
#include <log4cxx/logger.h>

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
 * A small module implementing the poor-man's version of the shortcut idea from Schraudolph98.
 *
 * \f$
 * y = tanh(Bx) - 0.5x
 * \f$
 */
struct poormans_shortcut_layer{

    poormans_shortcut_layer(unsigned int dim0, unsigned int dim1) :m_dim0(dim0), m_dim1(dim1){}
    unsigned int m_dim0, m_dim1;

    typedef boost::shared_ptr<Op>     op_ptr;
    boost::shared_ptr<ParameterInput>  m_B; 
    boost::shared_ptr<ParameterInput>  m_Bbias;
    boost::shared_ptr<ParameterInput>  m_Btbias;

    op_ptr m_y, m_f;

    void reset_weights(){
        if(m_B)
        {   // B: input_dim x dim0
            unsigned int input_dim = m_B->data().shape(0);
            float diff = 2.f*std::sqrt(6.f/(input_dim+m_dim1));   // tanh
            cuv::fill_rnd_uniform(m_B->data());
            m_B->data() *= 2*diff;
            m_B->data() -=   diff;
        }
        if(m_Bbias)
            m_Bbias->data() =  0.f;
        if(m_Btbias)
            m_Btbias->data() =  0.f;
    }
    op_ptr m_Bx;
    op_ptr m_tanh_Bx;
    op_ptr m_tanh_BTx;
    op_ptr encode(op_ptr inp, bool linear=false){
        op_group ("shortcut");

        if(!m_B){
            inp->visit(determine_shapes_visitor()); 
            unsigned int input_dim = inp->result()->shape[1];
            m_B.reset(new ParameterInput(cuv::extents[input_dim][m_dim1],"B"));
            m_Bbias.reset(new ParameterInput(cuv::extents[m_dim1], "Bbias"));
            m_Btbias.reset(new ParameterInput(cuv::extents[input_dim], "BtBias"));
        }
        m_Bx = mat_plus_vec(prod(inp, m_B), m_Bbias, 1);
        if(linear) {
            m_y = m_Bx;
        }else{
            m_tanh_Bx = tanh(m_Bx);
            m_y = m_tanh_Bx - 0.5 * m_Bx;
        }
        return label("poormans_shortcut_layer",m_y);
    }
    op_ptr decode(op_ptr enc, bool linear=false){
        if(linear)
            return mat_plus_vec(2.f*prod(enc, m_B, 'n','t'), m_Btbias, 1);
        op_ptr tmp = mat_plus_vec(2.f*prod(enc, m_B, 'n','t'), m_Btbias, 1);
        //return tanh(tmp) - 0.5f * tmp; // empirically worse classification-wise
        return tanh(tmp);
    }
    op_ptr jacobian_x(op_ptr enc){
        if(m_tanh_Bx){
            // this is a "real" shortcut layer
            return mat_times_vec(m_B, -0.5f + 1.f-pow(enc, 2.f), 1);
        }
        // this is a linear layer
        return m_B;
    }
    op_ptr schraudolph_regularizer(){
        // n/a for the poorman's version
        return op_ptr();
    }

    /**
     * Determine the parameters 
     */
    std::vector<Op*> params(bool unsupervised){
        using namespace boost::assign;
        std::vector<Op*> v;
        v += m_B.get();
        v += m_Bbias.get();
        if(unsupervised)
            v += m_Btbias.get();
        return v;
    };

};
/**
 * A small module implementing the shortcut idea from Schraudolph98.
 *
 * \f$
 * y = (tanh(Bx) + \alpha Bx + \beta) + Cx
 * \f$
 */
struct shortcut_layer{

    shortcut_layer(unsigned int dim0, unsigned int dim1) :m_dim0(dim0), m_dim1(dim1){}
    unsigned int m_dim0, m_dim1;

    typedef boost::shared_ptr<Op>     op_ptr;
    boost::shared_ptr<ParameterInput>  m_A; 
    boost::shared_ptr<ParameterInput>  m_B; 
    boost::shared_ptr<ParameterInput>  m_C; 
    boost::shared_ptr<ParameterInput>  m_alpha;
    boost::shared_ptr<ParameterInput>  m_beta;
    boost::shared_ptr<ParameterInput>  m_Bbias;
    boost::shared_ptr<ParameterInput>  m_Cbias;

    op_ptr m_y, m_f;

    void reset_weights(){
        if(m_B)
        {   // B: input_dim x dim0
            unsigned int input_dim = m_B->data().shape(0);
            float diff = 2.f*std::sqrt(6.f/(input_dim+m_dim0));   // tanh
            cuv::fill_rnd_uniform(m_B->data());
            m_B->data() *= 2*diff;
            m_B->data() -=   diff;
        }
        if(m_A)
        {   // A: dim0 x dim1
            float diff = 2.f*std::sqrt(6.f/(m_dim0+m_dim1));   // tanh
            cuv::fill_rnd_uniform(m_A->data());
            m_A->data() *= 2*diff;
            m_A->data() -=   diff;
        }
        {   // C: input_dim x dim1  (=shortcut)
            /*
             *unsigned int input_dim = m_C->data().shape(0);
             *float diff = 2.f*std::sqrt(6.f/(input_dim+m_dim1));   // tanh
             *cuv::fill_rnd_uniform(m_C->data());
             *m_C->data() *= 2*diff;
             *m_C->data() -=   diff;
             */
             m_C->data() = 0.f;
             m_Cbias->data() =  0.f;
        }
        if(m_alpha)
            m_alpha->data()  = -0.5f;
        if(m_beta)
            m_beta->data()   =  0.f;
        if(m_Bbias)
            m_Bbias->data() =  0.f;
    }
    op_ptr m_Bx;
    op_ptr m_tanh_Bx;
    op_ptr m_tanh_BTx;
    op_ptr encode(op_ptr inp, bool linear=false){
        op_group ("shortcut");
        if(linear){
            if(!m_C){
                inp->visit(determine_shapes_visitor()); 
                unsigned int input_dim = inp->result()->shape[1];
                m_C.reset(new ParameterInput(cuv::extents[input_dim][m_dim1],"C"));
                m_Cbias.reset(new ParameterInput(cuv::extents[m_dim1], "Cbias"));
            }
            m_y = mat_plus_vec(prod(inp, m_C), m_Cbias, 1);
        }else{
            if(!m_B){
                inp->visit(determine_shapes_visitor()); 
                unsigned int input_dim = inp->result()->shape[1];

                m_A.reset(new ParameterInput(cuv::extents[m_dim0][m_dim1],"A"));
                m_B.reset(new ParameterInput(cuv::extents[input_dim][m_dim0],"B"));
                m_C.reset(new ParameterInput(cuv::extents[input_dim][m_dim1],"C"));
                m_beta.reset(new ParameterInput(cuv::extents[m_dim0], "beta"));
                m_alpha.reset(new ParameterInput(cuv::extents[m_dim0], "alpha"));
                m_Bbias.reset(new ParameterInput(cuv::extents[m_dim0], "Bbias"));
                m_Cbias.reset(new ParameterInput(cuv::extents[m_dim1], "Cbias"));
            }
            m_Bx = mat_plus_vec(prod(inp, m_B), m_Bbias, 1);
            op_ptr Cx = mat_plus_vec(prod(inp, m_C), m_Cbias, 1);
            m_tanh_Bx = tanh(m_Bx);
            op_ptr alphaBx = mat_times_vec(m_Bx, m_alpha, 1);
            op_ptr alphaBx_plus_beta = mat_plus_vec(alphaBx, m_beta, 1);

            m_f = m_tanh_Bx + alphaBx_plus_beta;
            m_y = prod(m_f, m_A) + Cx;
        }

        return label("shortcut_layer",m_y);
    }
    op_ptr jacobian_x(op_ptr enc){
        if(m_alpha){
            // this is a "real" shortcut layer
            return mat_times_vec(m_B, 1.f-pow(enc, 2.f) + m_alpha, 1) + m_C;
        }
        // this is a linear layer
        return m_C;
    }
    op_ptr schraudolph_regularizer(){
        // f(x) and f'(x) should both be zero
        // derive w.r.t. alpha and beta
        if(!m_tanh_Bx)
            return op_ptr();
        op_ptr d_fenc_d_x = mat_plus_vec(1.f-pow(m_tanh_Bx,2.f), m_alpha, 1);
        op_ptr res = label("schraudolph_1l", mean(pow(m_f, 2.f)) + mean(pow(d_fenc_d_x, 2.f)));
        return res;
    }

    /**
     * Determine the parameters 
     */
    std::vector<Op*> params(){
        using namespace boost::assign;
        std::vector<Op*> v;
        if(m_B.get()){
            v += m_A.get(), m_B.get(), m_C.get(), m_Bbias.get(), m_Cbias.get();
            v += m_alpha.get(), m_beta.get();
        }else
            v +=  m_C.get(), m_Cbias.get();
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
        typedef poormans_shortcut_layer sc_t;
        // these are the parameters of the model
        boost::shared_ptr<sc_t> m_l0, m_l0a;
        boost::shared_ptr<sc_t> m_l1, m_l1a;

        unsigned int m_n_hidden0, m_n_hidden1;

    public:

        /// \f$ h := \sigma ( x W + b_h ) \f$
        virtual op_ptr  encode(op_ptr& inp){
            if(!m_encoded){
                inp->visit(determine_shapes_visitor());
                unsigned int input_dim = inp->result()->shape[1];
                m_l0.reset( new sc_t(m_n_hidden0, m_n_hidden0) );
                m_l0a.reset( new sc_t(m_n_hidden0, m_n_hidden1) );
                m_l1.reset( new sc_t(m_n_hidden0, m_n_hidden0) );
                m_l1a.reset( new sc_t(m_n_hidden0, input_dim) );

                //m_l0.reset( new sc_t(m_n_hidden0, m_n_hidden1) );
                //m_l1.reset( new sc_t(m_n_hidden0, input_dim) );
                m_encoded = m_l0a->encode(m_l0->encode(inp));
            }
            return m_encoded;
        }

        /// decode the encoded value given in `enc`
        virtual op_ptr  decode(op_ptr& enc){ 
            if(!m_decoded)
                //m_decoded = m_l1a->encode(m_l1->encode(enc), true);
                m_decoded = m_l0->decode(m_l0a->decode(enc), true);
            return m_decoded;
        }

        /**
         * Determine the parameters learned during pre-training
         * @overload
         */
        virtual std::vector<Op*> unsupervised_params(){
            std::vector<Op*> v0 = m_l0->params(true);
            std::vector<Op*> v0a = m_l0a->params(true);
            std::vector<Op*> v1 = m_l1->params(true);
            std::vector<Op*> v1a = m_l1a->params(true);
            v0.reserve(v0.size() + v1.size() + v0a.size() + v1a.size());
            v0.insert(v0.end(), v0a.begin(), v0a.end());
            //v0.insert(v0.end(), v1.begin(), v1.end());
            //v0.insert(v0.end(), v1a.begin(), v1a.end());
            return v0;
        };

        /**
         * Determine the parameters learned during fine-tuning
         * @overload
         */
        virtual std::vector<Op*> supervised_params(){
            std::vector<Op*> v0 = m_l0->params(false);
            std::vector<Op*> v0a = m_l0a->params(false);
            //std::vector<Op*> v1 = m_l1->supervised_params();
            v0.reserve(v0.size() + v0a.size());
            v0.insert(v0.end(), v0a.begin(), v0a.end());
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
             , m_n_hidden0(hidden_dim0)
             , m_n_hidden1(hidden_dim1)
    {
    }

        /**
         * initialize the weights with random numbers
         * @overload
         */
        virtual void reset_weights(){
            m_l0->reset_weights();
            m_l1->reset_weights();
            m_l0a->reset_weights();
            m_l1a->reset_weights();
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
