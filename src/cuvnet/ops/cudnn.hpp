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
                 * @param images nImages x nChannels x nPixels x nPixels
                 * @param filters nFilt x nFiltChannels x nFiltPix x nFiltPix
                 * TODO://fix this
                 */
               ConvolvecuDNN(result_t& images, result_t& filters, int m_padding_y=0, int m_padding_x=0, int m_ver_filt_stride=1, int m_hor_filt_stride=1)
                   :Op(2,1),
                    m_padding_x(m_padding_x),
                    m_padding_y(m_padding_y),
                    m_ver_filt_stride(m_ver_filt_stride),
                    m_hor_filt_stride(m_hor_filt_stride)
               {
                   add_param(0,images);
                   add_param(1,filters);
               }

                /**
                 * constructor.
                 *
                 * @param images nImages x nChannels x nPixels x nPixels
                 * @param filters nFilt x nFiltChannels x nFiltPix x nFiltPix
                 * @param bias nMaps
                 * TODO://fix this
                 */
               ConvolvecuDNN(result_t& images, result_t& filters, result_t& bias, int m_padding_y=0, int m_padding_x=0, int m_ver_filt_stride=1, int m_hor_filt_stride=1)
                   :Op(3,1),
                    m_padding_x(m_padding_x),
                    m_padding_y(m_padding_y),
                    m_ver_filt_stride(m_ver_filt_stride),
                    m_hor_filt_stride(m_hor_filt_stride)
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
     * cuDNN pooling
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
                //result_t::element_type m_result;
            public:
                PoolingcuDNN() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 * @subx pooling size
                 * @stridex distance between neighboring neurons in a bank
                 * @startx where to start implicitly, relative to left/top margin
                 * @param pt pooling type
                 */
                //TODO: fix this
                PoolingcuDNN(result_t& images, cuv::alex_conv::pool_type mode, int window_height, int window_width, int vertical_stride, int horizontal_stride)
                    :Op(1,1),
                    m_mode(mode),
                    m_window_height(window_height),
                    m_window_width(window_width),
                    m_vertical_stride(vertical_stride),
                    m_horizontal_stride(horizontal_stride)
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
