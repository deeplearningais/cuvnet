#ifndef __CONVOLUTIONAL_AUTO_ENCODER_HPP__
#     define __CONVOLUTIONAL_AUTO_ENCODER_HPP__

#include <cuvnet/ops.hpp>
#include "generic_auto_encoder.hpp"

using namespace cuvnet;
using boost::make_shared;


/**
 * implements a convolutional auto-encoder with untied weights
 *
 * note that this auto-encoder can easily learn the identity function,
 * so make sure that you regularize it in some way.
 * 
 * The most straight-forward regularization is to use the 
 * \c denoising_conv_auto_encoder instead of the plain one.
 */
template<class Regularizer>
class conv_auto_encoder
: virtual public generic_auto_encoder
, public Regularizer
{
    protected:

        boost::shared_ptr<Input> m_weights1, m_weights2;
        boost::shared_ptr<Input> m_bias1, m_bias2;

        unsigned int m_filter_size, m_n_filters, m_n_channels;

        /// \f$ h := \sigma ( x * W + b ) \f$
        virtual op_ptr  encode(op_ptr& inp){
            if(!m_weights1){
                inp->visit(determine_shapes_visitor()); 
                m_n_channels = inp->result()->shape[1];

                m_weights1.reset(new Input(cuv::extents[m_n_channels][m_filter_size*m_filter_size][m_n_filters], "weights1"));
                m_weights2.reset(new Input(cuv::extents[m_n_filters][m_filter_size*m_filter_size][m_n_channels], "weights2"));
                m_bias1.reset(new Input(cuv::extents[m_n_filters]));
                m_bias2.reset(new Input(cuv::extents[m_n_channels]));
            }
            return tanh(
                    mat_plus_vec(
                        convolve( 
                            reorder_for_conv(inp),
                            m_weights1,true),
                        m_bias1, 0)); 

        }
        /// \f$  h * W + b\f$
        virtual op_ptr  decode(op_ptr& enc){ 
            return reorder_from_conv(mat_plus_vec(convolve( enc, m_weights2, true), m_bias2, 0));
        }

    public:
        /**
         * Determine the parameters learned during pre-training
         * @overload
         */
        virtual std::vector<Op*> unsupervised_params(){
            return boost::assign::list_of(m_weights1.get())(m_weights2.get())(m_bias1.get())(m_bias2.get());
        };

        /**
         * Determine the parameters learned during fine-tuning
         * @overload
         */
        virtual std::vector<Op*> supervised_params(){
            return boost::assign::list_of(m_weights1.get())(m_bias1.get());
        };

        /**
         * @return weights operating on input
         */
        virtual boost::shared_ptr<Input> get_weights()const{
            return m_weights1;
        }

        /**
         * constructor
         *
         * @param binary if true, assume bernoulli-distributed inputs
         * @param filter_size the size of filters in the hidden layer
         * @param n_filters the number filters in the hidden layer
         */
        conv_auto_encoder(bool binary, unsigned int filter_size, unsigned int n_filters)
            : generic_auto_encoder(binary)
              , Regularizer(binary)
              , m_filter_size(filter_size)
              , m_n_filters(n_filters)
    {
    }

        /**
         * initialize the weights and biases with random numbers
         */
        virtual void reset_weights()
        {
            // initialize weights and biases
            //float diff = 4.f*std::sqrt(6.f/(m_n_channels*m_filter_size*m_filter_size + m_n_filters*m_filter_size*m_filter_size));
            
            // theano lenet tutorial style
            float fan_in = m_n_channels * m_filter_size * m_filter_size;
            float diff = std::sqrt(3.f/fan_in);

            cuv::fill_rnd_uniform(m_weights1->data());
            m_weights1->data() *= 2*diff;
            m_weights1->data() -=   diff;

            cuv::fill_rnd_uniform(m_weights2->data());
            m_weights2->data() *= 2*diff;
            m_weights2->data() -=   diff;

            m_bias1->data() =  0.f;
            //m_bias2->data() = m_binary ? 0.f : -0.5f;
            m_bias2->data() = m_binary ? 0.f : -0.0f;

        }
};




/**
 * Adds noise before calling the encoder of the conv_auto_encoder
 */
template<class Regularizer>
class denoising_conv_auto_encoder 
: public conv_auto_encoder<Regularizer>
{
    /// how much noise to add to inputs
    float m_noise;

    typedef typename conv_auto_encoder<Regularizer>::op_ptr op_ptr;
    using   typename conv_auto_encoder<Regularizer>::m_binary;

    public:

    /**
     * adds noise to the input before passing it on to \c conv_auto_encoder
     * @overload
     */
    virtual op_ptr  encode(op_ptr& inp){
        Op::op_ptr corrupt               = inp;
        if( m_binary && m_noise>0.f) corrupt =       zero_out(inp,m_noise);
        if(!m_binary && m_noise>0.f) corrupt = add_rnd_normal(inp,m_noise);
        return conv_auto_encoder<Regularizer>::encode(corrupt);
    }

    /**
     * constructor
     * 
     * @param binary if true, assumes inputs are bernoulli distributed
     * @param filter_size the size of filters in the hidden layer
     * @param n_filters the number filters in the hidden layer
     * @param noise if >0, add this much noise to input (type of noise depends on \c binary)
     */
    denoising_conv_auto_encoder(bool binary, unsigned int filter_size, unsigned int n_filters, float noise)
    :generic_auto_encoder(binary)
    ,conv_auto_encoder<Regularizer>(binary, filter_size, n_filters)
    ,m_noise(noise)
    {
    }
};

#endif /* __CONVOLUTIONAL_AUTO_ENCODER_HPP__ */
