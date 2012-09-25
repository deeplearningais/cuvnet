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
        }else{
            inp->visit(determine_shapes_visitor()); 
            unsigned int input_dim = inp->result()->shape[1];
            cuvAssert(m_weights->data().shape(0)==input_dim);
        }

        return tanh(mat_plus_vec(
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
    boost::shared_ptr<ParameterInput>  m_C; 
    boost::shared_ptr<ParameterInput>  m_alpha;
    boost::shared_ptr<ParameterInput>  m_beta;
    boost::shared_ptr<ParameterInput>  m_bias;

    op_ptr m_y;

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
             *unsigned int input_dim = m_C->data().shape(0);
             *float diff = 2.f*std::sqrt(6.f/(input_dim+m_dim));   // tanh
             *cuv::fill_rnd_uniform(m_C->data());
             *m_C->data() *= 2*diff;
             *m_C->data() -=   diff;
             */
             m_C->data() = 0.f;
        }
        m_alpha->data()  = -0.5f;
        m_beta->data()   =  0.f;
        m_bias->data() =  0.f;
    }
    op_ptr m_Bx;
    op_ptr m_tanh_Bx;
    op_ptr  encode(op_ptr inp){
        if(!m_B){
            inp->visit(determine_shapes_visitor()); 
            unsigned int input_dim = inp->result()->shape[1];

            m_B.reset(new ParameterInput(cuv::extents[input_dim][m_dim],"B"));
            m_C.reset(new ParameterInput(cuv::extents[input_dim][m_dim],"C"));
            m_beta.reset(new ParameterInput(cuv::extents[m_dim], "beta"));
            m_alpha.reset(new ParameterInput(cuv::extents[m_dim], "alpha"));
            m_bias.reset(new ParameterInput(cuv::extents[input_dim], "bias_y"));
        }

        m_Bx = prod(inp,   m_B);
        m_tanh_Bx = tanh(m_Bx);
        op_ptr alphaBx = mat_times_vec(m_Bx, m_alpha, 1);
        op_ptr alphaBx_plus_beta = mat_plus_vec(alphaBx, m_beta, 1);
        op_ptr Cx = prod(inp, m_C);
        m_y = m_tanh_Bx + alphaBx_plus_beta + Cx;


        return m_y;
    }
    op_ptr jacobian_x(op_ptr enc){
        return mat_times_vec(m_B, 1.f-pow(m_tanh_Bx, 2.f) + m_alpha, 1) + m_C;
    }

    op_ptr decode(op_ptr enc){
        // we assume that tanh(Bx) is to be reconstructed linearly, as in
        // normal auto-encoders.
        //
        // we therefore approximate 
        //
        //    tanh(Bx) + \alpha Bx    by 
        //    Bx+\alpha Bx            or shorter
        //    (1+\alpha) Bx           and get for the full inversion:
        //
        // \hat x = C'y + bias + B' [1/(1+alpha)] y
        op_ptr beta_and_C  =  mat_plus_vec(prod(enc, m_C, 'n','t'), m_bias, 1);
        op_ptr alpha_and_B =  prod(
                mat_times_vec(enc, pow(1.f + m_alpha, -1.f), 1),
                m_B, 'n', 't');
        return beta_and_C + alpha_and_B;
    }

    /**
     * Determine the parameters learned during pre-training
     * @overload
     */
    std::vector<Op*> unsupervised_params(){
        using namespace boost::assign;
        std::vector<Op*> v;
        v += m_B.get(), m_C.get(), m_bias.get(), m_beta.get();
        //v += m_alpha.get();
        return v;
    };

    /**
     * Determine the parameters learned during fine-tuning
     * @overload
     */
    std::vector<Op*> supervised_params(){
        using namespace boost::assign;
        std::vector<Op*> v;
        v += m_B.get(), m_C.get(), m_beta.get();
        //v += m_alpha.get();
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

    protected:
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
            return m_l0.decode(tanh(m_l1.decode(enc)));
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
