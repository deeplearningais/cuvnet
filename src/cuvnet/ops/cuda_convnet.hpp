#ifndef __OP_CUDA_CONVNET_HPP__
#     define __OP_CUDA_CONVNET_HPP__

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
     *  @ingroup CudaConvnetOps
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
                unsigned int m_nGroups;//number of groups of filters
                unsigned int m_partial_sum;
                int m_padding_start;
                int m_padding_size;
                int m_stride;
                bool m_symmetric_padding;
                cuv::tensor<int, matrix::memory_space_type> m_indices;
            public:
                inline int stride()const{return m_stride;}
                inline int padding_start()const{return m_padding_start;}
                inline int padding_size()const{return m_padding_size;}
                Convolve() :Op(2,1),m_symmetric_padding(true){} ///< for serialization

                /**
                 * constructor.
                 *
                 * @param images nChannels x nPixels x nImages
                 * @param filters nFiltChannels x nFiltPix x nFilt
                 * @param padding if true, pad image s.t. input and output shapes are equal.
                 * @padding_size if padding is true, it is used to pad images symmetrically(on both column and row)
                 * @stride distance between neighboring neurons in a bank
                 * @ngroups the number of groups that input and the filters are divided to
                 * @param partial_sum optimization parameter of alex' convolution routines. Good values are probably 4 or 8.
                 */
                Convolve(result_t& images, result_t& filters, bool padding, int padding_size, int stride, int ngroups, unsigned int partial_sum=4)
                    :Op(2,1)
                    ,m_nGroups(ngroups)
                    ,m_partial_sum(partial_sum)
                    ,m_padding_start(padding)
                    ,m_padding_size(padding_size)
                    ,m_stride(stride)
                {
                    add_param(0,images);
                    add_param(1,filters);
                }
                void fprop();
                void bprop();

                void _determine_shapes();
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

                void set_random_sparse(unsigned int n_filt_channels=0);

                /// @return true iff we use padding
                inline bool is_padded()const{
                    return m_padding_start != 0;
                }

                /// change the partial_sum parameter
                inline void set_partial_sum(int ps){
                    m_partial_sum = ps;
                }

                /// determines whether padding is same on both sides of the input
                inline void set_symmetric_padding(bool b){
                    m_symmetric_padding = b;
                }

                /// turn padding off (cannot be turned on again, since the value is overridden and not saved!)
                inline void disable_padding(){
                    m_padding_start = 0;
                    m_padding_size = 0;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_nGroups;
                        ar & m_partial_sum;
                        ar & m_padding_start;
                        ar & m_padding_size;
                        ar & m_stride;
                        if(version >= 1)
                            ar & m_indices;
                        if(version >= 2)
                            ar & m_symmetric_padding;
                    }
        };

    /**
     * Bed-of-nails subsampling (take every n-th value in the input maps)
     *
     * @ingroup CudaConvnetOps
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
                BedOfNails() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 * @param stridex interval for stepping
                 * @param startx start position for stepping
                 */
                BedOfNails(result_t& images, int stridex=2, int startx=0)
                    :Op(1,1),
                    m_startx(startx),
                    m_stridex(stridex)
                {
                    add_param(0,images);
                }
                inline unsigned int startx()const{return m_startx;}
                inline unsigned int stridex()const{return m_stridex;}
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
     * @ingroup CudaConvnetOps
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
                ResizeBilinear() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 * @param scale factor for bilinear scaling.
                 */
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
     * Response Norm across maps.
     *
     * @ingroup CudaConvnetOps
     */
    class ResponseNormalizationCrossMaps
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                int m_group_size;
                bool m_blocked;
                float m_add_scale, m_pow_scale;
                value_ptr m_orig_out;
                matrix m_denom;
            public:
                ResponseNormalizationCrossMaps() :Op(1,1){} ///< for serialization

                /**
                 * ctor.
                 * @param images input image function
                 * @param group_size number of maps around center to use for normalization
                 * @param add_scale add this to denominator
                 * @param pow_scale raise denominator to this power
                 * @param blocked ??? treat groups convolutionally / as non-overlapping blocks
                 */
                ResponseNormalizationCrossMaps(result_t& images, int group_size, float add_scale, float pow_scale, bool blocked)
                    :Op(1,1),
                    m_group_size(group_size),
                    m_blocked(blocked),
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
                        ar & m_group_size & m_add_scale & m_pow_scale & m_blocked;
                    }
        };
    /**
     * Response Norm.
     *
     * @ingroup CudaConvnetOps
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
                ResponseNormalization() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 * @param patch_size region size to normalize over
                 * @param add_scale this is added to stabilize denominator
                 * @param pow_scale the power to which input is raised
                 */
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
     * @ingroup CudaConvnetOps
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
                ContrastNormalization() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 * @param patch_size region size to normalize over
                 * @param add_scale this is added to stabilize denominator
                 * @param pow_scale the power to which input is raised
                 */
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
     * @ingroup CudaConvnetOps
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
                int m_startx;
                value_ptr m_result;
            public:
                LocalPooling() :Op(1,1), m_startx(0){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 * @subx pooling size
                 * @stridex distance between neighboring neurons in a bank
                 * @startx where to start implicitly, relative to left/top margin
                 * @param pt pooling type
                 */
                LocalPooling(result_t& images, int subsx, int stridex, cuv::alex_conv::pool_type pt, int startx=0)
                    :Op(1,1),
                    m_pooltype(pt),
                    m_subsx(subsx),
                    m_stridex(stridex),
                    m_startx(startx)
                {
                    add_param(0,images);
                }
                inline unsigned int subsx()const{return m_subsx;}
                inline unsigned int stridex()const{return m_stridex;}
                inline int startx()const{return m_startx;}
                inline void set_startx(int i){m_startx = i;}
                virtual void release_data();
                void fprop();
                void bprop();

                void _determine_shapes();
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_pooltype & m_subsx & m_stridex;
                        if(version>0){
                            ar & m_startx;
                        }
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
     * @ingroup CudaConvnetOps
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
                ReorderForConv() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 */
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
     * @ingroup CudaConvnetOps
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
                ReorderFromConv() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 */
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

BOOST_CLASS_VERSION(cuvnet::Convolve, 2)
BOOST_CLASS_VERSION(cuvnet::LocalPooling, 2)
#endif /* __OP_CUDA_CONVNET_HPP__ */
