#ifndef __OBJECT_DETECTION_HPP__
#     define __OBJECT_DETECTION_HPP__

#include <boost/assign.hpp>
#include <cuvnet/ops.hpp>

using namespace cuvnet;
using boost::make_shared;


/**
 * Implements a convolutional neural network for object detection.
 *
 * @ingroup models
 */
class obj_detector
{
    typedef boost::shared_ptr<Op> op_ptr;
    typedef boost::shared_ptr<ParameterInput> input_ptr;
    protected:
        input_ptr m_conv1_weights;
        input_ptr m_conv2_weights;
        input_ptr m_bias1, m_bias2, m_bias3;
        op_ptr m_loss;

        unsigned int m_n_channels;
        unsigned int m_filter_size1, m_filter_size2; 
        unsigned int m_n_filters1, m_n_filters2;


    public:
        op_ptr hl1, hl2, hl3;
        /**
         * initialize the LeNet with input and target
         *
         * Inputs must be square images.
         *
         * @param inp the input images (shape: batch size X number of color channels x number of pixels)
         * @param ignore a matrix which is 1 if the value
         * @param target the target labels (one-out-of-n coded)
         */
        virtual void init(op_ptr inp, op_ptr ignore, op_ptr target){
            inp->visit(determine_shapes_visitor()); 
            m_n_channels   = inp->result()->shape[1];
            int batchsize  = inp->result()->shape[0];

            int n_pix      = inp->result()->shape[2];
            int n_pix_x    = std::sqrt(n_pix);
            bool pad = true; // must be true, since we're subsampling `ignore' and `target' in parallel....
            //int n_pix_x2   = pad ? n_pix_x/2  : (n_pix_x  - m_filter_size1 / 2 - 1)/2;

            m_conv1_weights.reset(new ParameterInput(cuv::extents[m_n_channels][m_filter_size1*m_filter_size1][m_n_filters1], "conv_weights1"));
            m_conv2_weights.reset(new ParameterInput(cuv::extents[m_n_filters1][m_filter_size2*m_filter_size2][m_n_filters2], "conv_weights2"));

            m_bias1.reset(new ParameterInput(cuv::extents[m_n_filters1]));
            m_bias2.reset(new ParameterInput(cuv::extents[m_n_filters2]));

            hl1 =
                local_pool(
                        tanh(
                            mat_plus_vec(
                                convolve( 
                                    reorder_for_conv(inp),
                                    m_conv1_weights, pad),
                                m_bias1, 0)),
                        cuv::alex_conv::PT_MAX); 

            hl2 = 
                local_pool(
                        tanh(
                            mat_plus_vec(
                                convolve( 
                                    hl1,
                                    m_conv2_weights, pad),
                                m_bias2, 0)),
                        cuv::alex_conv::PT_MAX); 
            
            // pool target twice
            op_ptr subsampled_target = local_pool( 
                    local_pool( reorder_for_conv(target), cuv::alex_conv::PT_AVG),
                    cuv::alex_conv::PT_AVG);
            // pool ignore twice
            op_ptr subsampled_ignore = local_pool( 
                    local_pool( reorder_for_conv(ignore), cuv::alex_conv::PT_AVG),
                    cuv::alex_conv::PT_AVG);

            m_loss = mean(sum_to_vec(subsampled_ignore * pow(hl2 - subsampled_target, 2.f), 0));

            reset_weights();
        }

        /**
         * Determine the parameters learned during fine-tuning
         * @overload
         */
        virtual std::vector<Op*> params(){
            using namespace boost::assign;
            std::vector<Op*> params;
            params += m_conv1_weights.get();
            params += m_conv2_weights.get();
            params += m_bias1.get();
            params += m_bias2.get();
            return params;
        };

        op_ptr get_loss(){ return m_loss; }

        /**
         * constructor
         *
         * The parameter description below includes examples for the MNIST database
         *
         * @param filter_size1 the size of the first-layer filters (MNIST: 5)
         * @param n_filters1 the number of filters in the first layer (MNIST: 16)
         * @param filter_size2 the size of the first-layer filters (MNIST: 5)
         * @param n_filters2 the number of filters in the first layer (MNIST: 16)
         */
        obj_detector(int filter_size1, int n_filters1, int filter_size2, int n_filters2)
            : 
                m_filter_size1(filter_size1),
                m_filter_size2(filter_size2),
                m_n_filters1(n_filters1),
                m_n_filters2(n_filters2)
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
            } 

            m_bias1->data() =  0.f;
            m_bias2->data() =  0.f;
        }
};

#endif /* __OBJECT_DETECTION_HPP__ */