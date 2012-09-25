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
        boost::shared_ptr<ParameterInput>  m_weights0; ///< weights from input to 1st hidden layer
        boost::shared_ptr<ParameterInput>  m_weights1; ///< weights from input to 2nd hidden layer
        boost::shared_ptr<ParameterInput>  m_bias_h0;  ///< bias of 1st hidden layer
        boost::shared_ptr<ParameterInput>  m_bias_h1;  ///< bias of 2nd hidden layer
        boost::shared_ptr<ParameterInput>  m_bias_h0_;  ///< bias of 2nd hidden layer (decoding)
        boost::shared_ptr<ParameterInput>  m_bias_y;   ///< bias of output layer

        unsigned int m_hidden_dim0;  ///< size of 1st hidden layer
        unsigned int m_hidden_dim1;  ///< size of 2nd hidden layer

        op_ptr m_hl0; ///< the 1st hidden layer activation
        op_ptr m_hl1; ///< the 2nd hidden layer activation

    public:

        /// \f$ h := \sigma ( x W + b_h ) \f$
        virtual op_ptr  encode(op_ptr& inp){
            if(!m_weights0){
                inp->visit(determine_shapes_visitor()); 
                unsigned int input_dim = inp->result()->shape[1];

                m_weights0.reset(new ParameterInput(cuv::extents[input_dim][m_hidden_dim0],"weights0"));
                m_weights1.reset(new ParameterInput(cuv::extents[m_hidden_dim0][m_hidden_dim1],"weights1"));
                m_bias_h0.reset(new ParameterInput(cuv::extents[m_hidden_dim0],            "bias_h0"));
                m_bias_h0_.reset(new ParameterInput(cuv::extents[m_hidden_dim0],            "bias_h0_"));
                m_bias_h1 .reset(new ParameterInput(cuv::extents[m_hidden_dim1],            "bias_h1"));
                m_bias_y  .reset(new ParameterInput(cuv::extents[input_dim],                "bias_y"));
            }else{
                inp->visit(determine_shapes_visitor()); 
                unsigned int input_dim = inp->result()->shape[1];
                cuvAssert(m_weights0->data().shape(0)==input_dim);
            }

            op_ptr hl0_netin = mat_plus_vec(prod(inp,   m_weights0),m_bias_h0,1);
            m_hl0 = tanh(hl0_netin) + 0.5 * hl0_netin;
            //m_hl0 = tanh(hl0_netin);

            op_ptr hl1_netin = mat_plus_vec(prod(m_hl0, m_weights1),m_bias_h1,1);
            m_hl1 = tanh(hl1_netin) + 0.5 * hl1_netin;
            //m_hl1 = tanh(hl1_netin);

            return m_hl1;
        }
        /// decode the encoded value given in `enc`
        virtual op_ptr  decode(op_ptr& enc){ 
            op_ptr h0_ = tanh(mat_plus_vec(prod(enc, m_weights1, 'n','t'), m_bias_h0_, 1));
            op_ptr y_  =      mat_plus_vec(prod(h0_, m_weights0, 'n','t'), m_bias_y, 1);
            return y_;
        }

        /**
         * Determine the parameters learned during pre-training
         * @overload
         */
        virtual std::vector<Op*> unsupervised_params(){
            using namespace boost::assign;
            std::vector<Op*> v;
            v += m_weights0.get(), m_weights1.get(), m_bias_h0.get(), m_bias_h1.get(), m_bias_h0_.get(), m_bias_y.get();
            return v;
        };

        /**
         * Determine the parameters learned during fine-tuning
         * @overload
         */
        virtual std::vector<Op*> supervised_params(){
            using namespace boost::assign;
            std::vector<Op*> v;
            v += m_weights0.get(), m_weights1.get(), m_bias_h0.get(), m_bias_h1.get();
            return v;
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
             ,m_hidden_dim0(hidden_dim0)
             ,m_hidden_dim1(hidden_dim1)
    {
    }

        /**
         * initialize the weights with random numbers
         * @overload
         */
        virtual void reset_weights(){
            {
                unsigned int input_dim = m_weights0->data().shape(0);
                //float diff = 4.f*std::sqrt(6.f/(input_dim+m_hidden_dim0)); // logistic
                float diff = 2.f*std::sqrt(6.f/(input_dim+m_hidden_dim0));   // tanh
                cuv::fill_rnd_uniform(m_weights0->data());
                m_weights0->data() *= 2*diff;
                m_weights0->data() -=   diff;
            }
            {
                unsigned int input_dim = m_weights1->data().shape(0);
                //float diff = 4.f*std::sqrt(6.f/(input_dim+m_hidden_dim1)); // logistic
                float diff = 2.f*std::sqrt(6.f/(input_dim+m_hidden_dim1));  // tanh
                cuv::fill_rnd_uniform(m_weights1->data());
                m_weights1->data() *= 2*diff;
                m_weights1->data() -=   diff;
            }
            m_bias_h0->data()   = 0.f;
            m_bias_h1->data()   = 0.f;
            m_bias_h0_->data()  = 0.f;
            m_bias_y->data()   = 0.f;
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
