#ifndef __OBJECT_DETECTION_HPP__
#     define __OBJECT_DETECTION_HPP__

#include <boost/assign.hpp>
#include <cuvnet/ops.hpp>
#include <tools/normalization.hpp>

using namespace cuvnet;
using boost::make_shared;


/**
 * Implements a convolutional neural network for object detection.
 *
 * @ingroup models
 */
class obj_detector
{
    public:
        typedef boost::shared_ptr<Op> op_ptr;
        typedef boost::shared_ptr<ParameterInput> input_ptr;
        typedef boost::shared_ptr<Sink> sink_ptr;
    private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version){
                ar & m_conv1_weights & m_conv2_weights & m_conv3_weights;
                ar & m_bias1 & m_bias2 & m_bias3;
                ar & m_loss;
                ar & m_n_channels;
                ar & m_filter_size1 & m_filter_size2 & m_filter_size3;
                ar & m_n_filters1 & m_n_filters2 & m_n_filters3;
                ar & hl3; // needed for valid_shape_info
            }
    public:
        input_ptr m_conv1_weights;
        input_ptr m_conv2_weights;
        input_ptr m_conv3_weights;
        input_ptr m_bias1, m_bias2, m_bias3;
        sink_ptr m_output;
        op_ptr m_loss;

        unsigned int m_n_channels;
        unsigned int m_filter_size1, m_filter_size2, m_filter_size3; 
        unsigned int m_n_filters1, m_n_filters2, m_n_filters3;

        op_ptr actfunc(op_ptr f){
            //return tanh(f);
            return rectified_linear(f);
            //return response_normalization(rectified_linear(f), 7, 1.f, 1.f);
            //return contrast_normalization(tanh(f), 7, 1.f, 1.f);
        }

        op_ptr conv_bias(op_ptr images, op_ptr weights, op_ptr bias = op_ptr()){
            int partialSum = 16;
            images->visit(determine_shapes_visitor()); 
            if((images->result()->shape[1] * images->result()->shape[2]) % partialSum != 0)
                partialSum = 8;
            if((images->result()->shape[1] * images->result()->shape[2]) % partialSum != 0)
                partialSum = 4;
            if((images->result()->shape[1] * images->result()->shape[2]) % partialSum != 0)
                partialSum = 2;
            if((images->result()->shape[1] * images->result()->shape[2]) % partialSum != 0)
                partialSum = 1;
            
            bool pad = false;
            if(!bias)
                return convolve(images, weights, pad, partialSum);
            else
                return mat_plus_vec(convolve(images, weights, pad, partialSum), bias, 0);
        }

        op_ptr pool(op_ptr images){
            return local_pool(images, cuv::alex_conv::PT_MAX);
            //return sqrt(local_pool(pow(images, 2.f), cuv::alex_conv::PT_AVG));
        }

    public:
        op_ptr hl1, hl2, hl3;
        /**
         * initialize the LeNet with input and target
         *
         * Inputs must be square images.
         *
         * @param inp the input images (shape: batch size X number of color channels x number of pixels)
         * @param n_omaps the number of output maps
         */
        virtual void init(op_ptr inp, unsigned int n_omaps){
            inp->visit(determine_shapes_visitor()); 
            unsigned int bs= inp->result()->shape[0];
            m_n_channels   = inp->result()->shape[1];

            m_n_filters3   = n_omaps;

            m_conv1_weights.reset(new ParameterInput(cuv::extents[m_n_channels][m_filter_size1*m_filter_size1][m_n_filters1], "conv_weights1"));
            m_conv2_weights.reset(new ParameterInput(cuv::extents[m_n_filters1][m_filter_size2*m_filter_size2][m_n_filters2], "conv_weights2"));
            m_conv3_weights.reset(new ParameterInput(cuv::extents[m_n_filters2][m_filter_size3*m_filter_size3][m_n_filters3], "conv_weights3"));

            m_bias1.reset(new ParameterInput(cuv::extents[m_n_filters1], "bias1"));
            m_bias2.reset(new ParameterInput(cuv::extents[m_n_filters2], "bias2"));
            m_bias3.reset(new ParameterInput(cuv::extents[m_n_filters3], "bias3"));

            op_ptr input_pool0 = reorder_for_conv(inp);
            hl1 = actfunc(conv_bias(input_pool0,   m_conv1_weights, m_bias1));
            hl2 = actfunc(conv_bias(pool(hl1), m_conv2_weights, m_bias2));
            hl3 =         conv_bias(pool(hl2), m_conv3_weights, m_bias3);

            m_output = sink("output", hl3);

            hl3->visit(determine_shapes_visitor());
            unsigned int out_h = hl3->result()->shape[1];
            unsigned int out_w = hl3->result()->shape[2];
            std::cout << "out_h:" << out_h << " out_w:" << out_w << std::endl;

            boost::shared_ptr<ParameterInput> ignore(
                    new ParameterInput(cuv::extents[bs][n_omaps][out_h][out_w],"ignore"));
            boost::shared_ptr<ParameterInput> target(
                    new ParameterInput(cuv::extents[bs][n_omaps][out_h][out_w],"target"));

            op_ptr subsampled_target = reorder_for_conv(target);
            op_ptr subsampled_ignore = reorder_for_conv(ignore);
                    ;

            // batch is in the dimension with index 2
            //m_loss = mean(subsampled_ignore * square(hl3 - subsampled_target));
            //m_loss = mean(subsampled_ignore * neg_log_cross_entropy_of_logistic(subsampled_target, hl3));
            //m_loss = mean(subsampled_ignore * epsilon_insensitive_loss(0.05f, subsampled_target, logistic(hl3)));
            //m_loss = mean(subsampled_ignore * epsilon_insensitive_loss(0.05f, subsampled_target, hl3));
            m_loss = mean(subsampled_ignore * squared_hinge_loss(subsampled_target, hl3));

            m_f2.reset(new F2Measure(
                        sink("sst", subsampled_target)->result(),
                        sink("hl3", hl3)->result(),
                        sink("ign", subsampled_ignore)->result(),
                        0.f));

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
            params += m_conv3_weights.get();
            params += m_bias1.get();
            params += m_bias2.get();
            params += m_bias3.get();
            return params;
        };

        op_ptr get_loss(){ return m_loss; }

        op_ptr m_f2;
        op_ptr get_f2(){ 
            return m_f2; 
        }

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
                m_filter_size3(filter_size2),
                m_n_filters1(n_filters1),
                m_n_filters2(n_filters2),
                m_n_filters3(n_filters2)
    {
    }

        /** Default constructor for serialization purposes */
        obj_detector() { }

        /**
         * project all weights to the allowed set.
         */
        void project_to_allowed_region(){
            project_to_unit_ball(m_conv1_weights->data(), 2, 5.f);
            project_to_unit_ball(m_conv2_weights->data(), 2, 5.f);
            project_to_unit_ball(m_conv3_weights->data(), 2, 5.f);
        }

        /**
         * initialize the weights and biases with random numbers
         */
        virtual void reset_weights()
        {
            {
                float fan_in  = m_n_channels * m_filter_size1 * m_filter_size1;
                float fan_out = m_n_filters1 * m_filter_size2 * m_filter_size2;
                float diff = std::sqrt(6.f/(fan_in+fan_out));
                //float diff = std::sqrt(3.f/(fan_in));
                //diff *= .1f;
    
                cuv::fill_rnd_uniform(m_conv1_weights->data());
                m_conv1_weights->data() *= 2*diff;
                m_conv1_weights->data() -=   diff;
            } 
            {
                float fan_in = m_n_filters1 * m_filter_size2 * m_filter_size2;
                float fan_out = m_n_filters2 * m_filter_size3 * m_filter_size3;
                float diff = std::sqrt(6.f/(fan_in + fan_out));
                //float diff = std::sqrt(3.f/(fan_in));
                //diff *= .1f;
    
                cuv::fill_rnd_uniform(m_conv2_weights->data());
                m_conv2_weights->data() *= 2*diff;
                m_conv2_weights->data() -=   diff;
            } 
            {
                float fan_in = m_n_filters2 * m_filter_size3 * m_filter_size3;
                float fan_out = 1 * 1 * 1;
                float diff = std::sqrt(6.f/(fan_in + fan_out));
                //float diff = std::sqrt(3.f/(fan_in));
                //diff *= .1f;
    
                cuv::fill_rnd_uniform(m_conv3_weights->data());
                m_conv3_weights->data() *= 2*diff;
                m_conv3_weights->data() -=   diff;
                //m_conv3_weights->data() = 0.f;
            } 

            //m_conv1_weights->set_learnrate_factor ( 1.f / 3.f );
            //m_conv1_weights->m_learnrate_factor = 1.f / (m_filter_size1 * m_filter_size1);
            //m_conv2_weights->m_learnrate_factor = 1.f / (m_filter_size2 * m_filter_size2);
            //m_conv3_weights->m_learnrate_factor = 1.f / (m_filter_size3 * m_filter_size3);

            m_bias1->data() =  0.f;
            m_bias2->data() =  0.f;
            m_bias3->data() =  0.f;
            //m_bias1->m_learnrate_factor = 1.f / (m_filter_size1 * m_filter_size1);
            //m_bias2->m_learnrate_factor = 1.f / (m_filter_size2 * m_filter_size2);
            //m_bias3->m_learnrate_factor = 1.f / (m_filter_size3 * m_filter_size3);
            //m_bias1->set_weight_decay_factor ( 0.f);
            //m_bias2->set_weight_decay_factor ( 0.f);
            //m_bias3->set_weight_decay_factor ( 0.f);
        }
};

#endif /* __OBJECT_DETECTION_HPP__ */
