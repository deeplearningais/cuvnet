#ifndef __OP_CONVOLVE_HPP__
#     define __OP_CONVOLVE_HPP__

#include <cmath>
#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <log4cxx/logger.h>
namespace cuvnet
{

    /**
     * convolve a set of images using a set of filters.
     *
     * This is the "neural net" type convolution, where filters are 3D and
     * results are summed over input channels.
     *
     * First param, the images, must have shape 
     *
     *  nChannels x nPixels x nImages
     *
     * while filters have shape
     *  
     *  nFiltChannels x nFiltPix x nFilt
     *
     *  @ingroup Ops
     *
     */
    class Convolve
        :public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_nGroups;
                unsigned int m_partial_sum;
                int m_padding_start;
            public:
                Convolve() :Op(2,1){} // for serialization

                /**
                 * constructor.
                 *
                 * @param images nChannels x nPixels x nImages
                 * @param filters nFiltChannels x nFiltPix x nFilt
                 * @param partial_sum optimization parameter of alex' convolution routines. Good values are probably 4 or 8.
                 */
                Convolve(result_t& images, result_t& filters, bool padding, unsigned int partial_sum=4)
                    :Op(2,1)
                    ,m_nGroups(1)
                    ,m_partial_sum(partial_sum)
                    ,m_padding_start(padding)
                {
                    add_param(0,images);
                    add_param(1,filters);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

                inline bool is_padded()const{
                    return m_padding_start != 0;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_nGroups;
                        ar & m_partial_sum;
                        ar & m_padding_start;
                    }
        };

    /**
     * Bed-of-nails subsampling (take every n-th value in the input maps)
     *
     * @ingroup Ops
     */
    class BedOfNails
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_startx, m_stridex;
            public:
                BedOfNails() :Op(1,1){} // for serialization
                BedOfNails(result_t& images, int stridex=2, int startx=0)
                    :Op(1,1),
                    m_startx(startx),
                    m_stridex(stridex)
                {
                    add_param(0,images);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_stridex & m_startx;
                    }
        };

    /**
     * Resize Bilinear
     *
     * @ingroup Ops
     */
    class ResizeBilinear
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                float m_scale;
            public:
                ResizeBilinear() :Op(1,1){} // for serialization
                ResizeBilinear(result_t& images, float scale)
                    :Op(1,1),
                    m_scale(scale)
                {
                    add_param(0,images);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scale;
                    }
        };

    /**
     * Separable Filter.
     *
     * @ingroup Ops
     */
    class SeparableFilter
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                matrix m_kernel;
            public:
                SeparableFilter() :Op(1,1){} // for serialization
                SeparableFilter(result_t& images, const matrix& kernel)
                    :Op(1,1),
                    m_kernel(kernel)
                {
                    add_param(0,images);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_kernel;
                    }
        };

    /**
     * Response Norm.
     *
     * @ingroup Ops
     */
    class ResponseNormalization
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                int m_patch_size;
                float m_add_scale, m_pow_scale;
                value_ptr m_orig_out;
                matrix m_denom;
            public:
                ResponseNormalization() :Op(1,1){} // for serialization
                ResponseNormalization(result_t& images, int patch_size, float add_scale, float pow_scale)
                    :Op(1,1),
                    m_patch_size(patch_size),
                    m_add_scale(add_scale),
                    m_pow_scale(pow_scale)
                {
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
                        ar & m_patch_size & m_add_scale & m_pow_scale;
                    }
        };

    /**
     * Contrast Normalization
     *
     * @ingroup Ops
     */
    class ContrastNormalization
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                int m_patch_size;
                float m_add_scale, m_pow_scale;
                value_ptr m_orig_out;
                matrix m_denom;
                matrix m_meandiffs;
            public:
                ContrastNormalization() :Op(1,1){} // for serialization
                ContrastNormalization(result_t& images, int patch_size, float add_scale, float pow_scale)
                    :Op(1,1),
                    m_patch_size(patch_size),
                    m_add_scale(add_scale),
                    m_pow_scale(pow_scale)
                {
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
                        ar & m_patch_size & m_add_scale & m_pow_scale;
                    }
        };

    /**
     * Maximum pooling or subsampling of image regions.
     *
     * @ingroup Ops
     */
    class LocalPooling
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                cuv::alex_conv::pool_type m_pooltype;
                unsigned int m_subsx, m_stridex;
                value_ptr m_result;
            public:
                LocalPooling() :Op(1,1){} // for serialization
                LocalPooling(result_t& images, cuv::alex_conv::pool_type pt)
                    :Op(1,1),
                    m_pooltype(pt),
                    m_subsx(2),
                    m_stridex(2)
                {
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
                        ar & m_pooltype & m_subsx & m_stridex;
                    }
        };

    /**
     * converts the (more natural) memory order nImg x nChann x nPixY x nPixX to the
     * order required by Alex' convolution routines.
     * 
     * ...which is nChann x nPixY x nPixX x nImg
     *
     * In bprop(), it does the opposite operation. This
     * is quite cheap compared to convolution itself.
     * (less than 1% for large images)
     *
     * @ingroup Ops
     *
     */
    class ReorderForConv
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
            public:
                ReorderForConv() :Op(1,1){} // for serialization
                ReorderForConv(result_t& images)
                    :Op(1,1)
                {
                    add_param(0,images);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };

    /**
     * Does the opposite of \c ReorderForConv.
     *
     * @ingroup Ops
     */
    class ReorderFromConv
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
            public:
                ReorderFromConv() :Op(1,1){} // for serialization
                ReorderFromConv(result_t& images)
                    :Op(1,1)
                {
                    add_param(0,images);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };

}
#endif /* __OP_CONVOLVE_HPP__ */
