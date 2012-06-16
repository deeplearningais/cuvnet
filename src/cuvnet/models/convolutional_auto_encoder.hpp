#ifndef __CONVOLUTIONAL_AUTO_ENCODER_HPP__
#     define __CONVOLUTIONAL_AUTO_ENCODER_HPP__

#include <cuvnet/ops.hpp>
#include "generic_auto_encoder.hpp"

using namespace cuvnet;
using boost::make_shared;

template<class Regularizer>
class conv_auto_encoder
: virtual public generic_auto_encoder
, public Regularizer
{
    private:

        boost::shared_ptr<Input> m_weights1, m_weights2;

        unsigned int m_filter_size, m_n_filters, m_n_channels;

    /// \f$ h := \sigma ( x * W  ) \f$
    virtual op_ptr  encode(op_ptr& inp){
        if(!m_weights1){
            inp->visit(determine_shapes_visitor()); 
            unsigned int m_n_channels = inp->result()->shape[1];
            
            m_weights1.reset(new Input(cuv::extents[m_n_channels][m_filter_size*m_filter_size][m_n_filters], "weights1"));
            m_weights2.reset(new Input(cuv::extents[m_n_filters][m_filter_size*m_filter_size][m_n_channels], "weights2"));
        }
        //m_input = reorder__conv(inp);
        return logistic(
                convolve( 
                    reorder_for_conv(m_input),
                    m_weights1,true)); 

    }
    /// \f$  h * W \f$
    virtual op_ptr  decode(op_ptr& enc){ 
        return reorder_from_conv(
                    convolve( enc, m_weights2, true));
    }

    /**
     * Determine the parameters learned during pre-training
     * @overload
     */
    virtual std::vector<Op*> unsupervised_params(){
        return boost::assign::list_of(m_weights1.get())(m_weights2.get());
    };

    /**
     * Determine the parameters learned during fine-tuning
     * @overload
     */
    virtual std::vector<Op*> supervised_params(){
        return boost::assign::list_of(m_weights1.get());
    };

    public:

    conv_auto_encoder(bool binary, unsigned int filter_size, unsigned int n_filters)
    : generic_auto_encoder(binary)
    , Regularizer(binary)
    , m_filter_size(filter_size)
    , m_n_filters(n_filters)
    {
    }

    virtual void reset_weights()
    {
        // initialize weights and biases
        float diff = 4.f*std::sqrt(6.f/(m_n_channels*m_filter_size*m_filter_size + m_n_filters*m_filter_size*m_filter_size));

        cuv::fill_rnd_uniform(m_weights1->data());
        m_weights1->data() *= 2*diff;
        m_weights1->data() -=   diff;

        cuv::fill_rnd_uniform(m_weights2->data());
        m_weights2->data() *= 2*diff;
        m_weights2->data() -=   diff;

    }
};

#endif /* __CONVOLUTIONAL_AUTO_ENCODER_HPP__ */
