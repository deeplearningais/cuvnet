#ifndef __LENET_HPP__
#     define __LENET_HPP__

#include <boost/assign.hpp>
#include <cuvnet/ops.hpp>
#include "logistic_regression.hpp"
#include "linear_regression.hpp"

using namespace cuvnet;
using boost::make_shared;


/**
 * Implements a convolutional neural network for classification as proposed by Yann LeCun.
 *
 * The input for the network are 3-dimensional arrays, where
 * the first dimension is the batch size, the second is the
 * number of color channels, and the third is the number of
 * pixels in one color channel. The images are square.
 *
 * The network consists of 
 * - A first convolution of the image 
 * - A first maximum-pooling operation
 * - A second convolution of the image
 * - A second maximum-pooling operation
 * - A fully connected MLP-like hidden layer
 * - A logistic regression module for classification.
 *
 * @example lenet.cpp
 *
 * @ingroup models
 */
class lenet
{
    typedef boost::shared_ptr<Op> op_ptr;
    typedef boost::shared_ptr<Input> input_ptr;
    typedef logistic_regression regression_type;
    //typedef linear_regression regression_type;
    protected:
        input_ptr m_conv1_weights;
        input_ptr m_conv2_weights;
        input_ptr m_weights3;
        input_ptr m_bias1, m_bias2, m_bias3;

        boost::shared_ptr<regression_type> m_logreg;

        unsigned int m_n_channels;
        unsigned int m_filter_size1, m_filter_size2; 
        unsigned int m_n_filters1, m_n_filters2;
        unsigned int m_n_hiddens;


    public:
        op_ptr hl1, hl2, hl3;
        virtual void init(op_ptr inp, op_ptr target){
            inp->visit(determine_shapes_visitor()); 
            m_n_channels   = inp->result()->shape[1];
            int batchsize  = inp->result()->shape[0];

            int n_pix      = inp->result()->shape[2];
            int n_pix_x    = std::sqrt(n_pix);
            int n_pix_x2   = (n_pix_x  - m_filter_size1 / 2 - 1)/2;
            int n_pix_x3   = (n_pix_x2 - m_filter_size2 / 2 - 1)/2;

            m_conv1_weights.reset(new Input(cuv::extents[m_n_channels][m_filter_size1*m_filter_size1][m_n_filters1], "conv_weights1"));
            m_conv2_weights.reset(new Input(cuv::extents[m_n_filters1][m_filter_size2*m_filter_size2][m_n_filters2], "conv_weights2"));

            m_weights3.reset(new Input(cuv::extents[n_pix_x3*n_pix_x3*m_n_filters2][m_n_hiddens], "weights3"));

            m_bias1.reset(new Input(cuv::extents[m_n_filters1]));
            m_bias2.reset(new Input(cuv::extents[m_n_filters2]));
            m_bias3.reset(new Input(cuv::extents[m_n_hiddens]));

            hl1 =
                local_pool(
                        tanh(
                            mat_plus_vec(
                                convolve( 
                                    reorder_for_conv(inp),
                                    m_conv1_weights, false),
                                m_bias1, 0)),
                        cuv::alex_conv::PT_MAX); 

            hl2 = 
                local_pool(
                        tanh(
                            mat_plus_vec(
                                convolve( 
                                    hl1,
                                    m_conv2_weights, false),
                                m_bias2, 0)),
                        cuv::alex_conv::PT_MAX); 
            hl2 = reorder_from_conv(hl2);
            hl2 = reshape(hl2, cuv::extents[batchsize][-1]);
            hl3 =
                tanh(
                mat_plus_vec(
                    prod(hl2, m_weights3),
                    m_bias3,1));

            m_logreg = boost::make_shared<regression_type>(hl3, target);
            reset_weights();
        }

        /**
         * Determine the parameters learned during fine-tuning
         * @overload
         */
        virtual std::vector<Op*> params(){
            using namespace boost::assign;
            std::vector<Op*> params = m_logreg->params();
            params += m_conv1_weights.get();
            params += m_conv2_weights.get();
            params += m_weights3.get();
            params += m_bias1.get();
            params += m_bias2.get();
            params += m_bias3.get();
            return params;
        };

        op_ptr classification_error(){ return m_logreg->classification_error(); }
        op_ptr get_loss(){ return m_logreg->get_loss(); }

        /**
         * constructor
         *
         */
        lenet(int filter_size1, int n_filters1, int filter_size2, int n_filters2, int n_hiddens)
            : 
                m_filter_size1(filter_size1),
                m_filter_size2(filter_size2),
                m_n_filters1(n_filters1),
                m_n_filters2(n_filters2),
                m_n_hiddens(n_hiddens)
    {
    }

        /**
         * initialize the weights and biases with random numbers
         */
        virtual void reset_weights()
        {
            {
                float fan_in = m_n_channels * m_filter_size1 * m_filter_size1;
                float diff = std::sqrt(3.f/fan_in);
    
                cuv::fill_rnd_uniform(m_conv1_weights->data());
                m_conv1_weights->data() *= 2*diff;
                m_conv1_weights->data() -=   diff;
            } {
                float fan_in = m_n_filters1 * m_filter_size2 * m_filter_size2;
                float diff = std::sqrt(3.f/fan_in);
    
                cuv::fill_rnd_uniform(m_conv2_weights->data());
                m_conv2_weights->data() *= 2*diff;
                m_conv2_weights->data() -=   diff;
            } {
                float wnorm = m_weights3->data().shape(0)
                    +         m_weights3->data().shape(1);
                float diff = std::sqrt(6.f/wnorm);
                //diff *= 4.f; // for logistic activation function "only"
                cuv::fill_rnd_uniform(m_weights3->data());
                m_weights3->data() *= diff*2.f;
                m_weights3->data() -= diff;
            }

            m_bias1->data() =  0.f;
            m_bias2->data() =  0.f;
            m_bias3->data() =  0.f;

            m_logreg->reset_weights();
        }
};

#endif /* __LENET_HPP__ */
