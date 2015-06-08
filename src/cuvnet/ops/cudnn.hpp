#ifndef __OP_CUDNN_HPP__
#     define __OP_CUDNN_HPP__

#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <log4cxx/logger.h>
namespace cuvnet
{
    struct cudnn_state;

    /**
     * Convolution via cuDNN library.
     *
     * @ingroup CuDNNOps
     */
    class ConvolvecuDNN
       :public Op{
           public:
               typedef Op::value_type    value_type;
               typedef Op::op_ptr        op_ptr;
               typedef Op::value_ptr     value_ptr;
               typedef Op::param_t       param_t;
               typedef Op::result_t      result_t;
           private:
       		int m_padding_x = 0;
       		int m_padding_y = 0;
       		int m_ver_filt_stride = 1;
       		int m_hor_filt_stride = 1;
            boost::shared_ptr<cudnn_state> m_state;
            friend struct cudnn_state;
           public:
               ConvolvecuDNN() :Op(2,1){} ///< for serialization
                /**
                 * constructor.
                 *
                 * @param images nImages x nMaps x image height x image width
                 * @param filters nFilters x nMaps x filter height x filter width
                 * @param padding_y zero-padding height
                 * @param padding_x zero-padding width
                 * @param ver_filt_stride vertical filter stride
                 * @param hor_filt_stride horizontal filter stride
                 */
               ConvolvecuDNN(result_t& images, result_t& filters, int padding_y=0, int padding_x=0, int ver_filt_stride=1, int hor_filt_stride=1)
                   :Op(2,1),
                    m_padding_x(padding_x),
                    m_padding_y(padding_y),
                    m_ver_filt_stride(ver_filt_stride),
                    m_hor_filt_stride(hor_filt_stride)
               {
                   add_param(0,images);
                   add_param(1,filters);
               }

                /**
                 * constructor.
                 *
                 * @param images nImages x nMaps x image height x image width
                 * @param filters nFilters x nMaps x filter height x filter width
                 * @param bias nImages x nMaps x height x width
                 * @param padding_y zero-padding height
                 * @param padding_x zero-padding width
                 * @param ver_filt_stride vertical filter stride
                 * @param hor_filt_stride horizontal filter stride
                 */
               ConvolvecuDNN(result_t& images, result_t& filters, result_t& bias, int padding_y=0, int padding_x=0, int ver_filt_stride=1, int hor_filt_stride=1)
                   :Op(3,1),
                    m_padding_x(padding_x),
                    m_padding_y(padding_y),
                    m_ver_filt_stride(ver_filt_stride),
                    m_hor_filt_stride(hor_filt_stride)
               {
                   add_param(0,images);
                   add_param(1,filters);
                   add_param(2,bias);
               }


               void fprop();
               void bprop();
               void _determine_shapes();
               void _graphviz_node_desc(detail::graphviz_node& desc)const;

           private:
               friend class boost::serialization::access;
               template<class Archive>
                   void serialize(Archive& ar, const unsigned int version){
                       ar & boost::serialization::base_object<Op>(*this);
                       ar & m_padding_x;
                       ar & m_padding_y;
                       ar & m_ver_filt_stride;
                       ar & m_hor_filt_stride;
                   }

       };

    struct cudnn_pooling_state;
    /**
     * Pooling via cuDNN library.
     *
     * @ingroup CuDNNOps
     */
    class PoolingcuDNN
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                cuv::alex_conv::pool_type m_mode;
                int m_window_height;
                int m_window_width;
                int m_vertical_stride;
                int m_horizontal_stride;
                int m_vertical_pad;
                int m_horizontal_pad;
                value_ptr m_result;
                boost::shared_ptr<cudnn_pooling_state> m_state;
                friend struct cudnn_pooling_state;
            public:
                PoolingcuDNN() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images nImages x nMaps x height x width
                 * @param mode pooling mode
                 * @param window_height height of the pooling window
                 * @param window_width width of the pooling window
                 * @param vertical_stride pooling vertical stride
                 * @param horizontal_stride pooling horizontal stride
                 */
                PoolingcuDNN(result_t& images, cuv::alex_conv::pool_type mode, int window_height, int window_width, int vertical_stride, int horizontal_stride, int vertical_padding=0, int horizontal_padding=0)
                    :Op(1,1),
                    m_mode(mode),
                    m_window_height(window_height),
                    m_window_width(window_width),
                    m_vertical_stride(vertical_stride),
                    m_horizontal_stride(horizontal_stride),
                	m_vertical_pad(vertical_padding),
                	m_horizontal_pad(horizontal_padding)
                {
                	if(m_window_height == 3){
                        m_vertical_pad = m_window_height % 2 == 0 ? 0 : m_window_height / 2;
                        m_horizontal_pad = m_window_width % 2 == 0 ? 0 : m_window_width / 2;
                    }else{
                        m_vertical_pad = 0;
                        m_horizontal_pad = 0;
                    }
                    add_param(0,images);
                }
                virtual void release_data();
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_mode;
                        ar & m_window_height;
                        ar & m_window_width;
                        ar & m_vertical_stride;
                        ar & m_horizontal_stride;
                        ar & m_vertical_pad;
                        ar & m_horizontal_pad;
                    }
        };
}
#endif /* __OP_CUDNN_HPP__ */
