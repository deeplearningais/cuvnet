#ifndef __OP_CONVOLVE_HPP__
#     define __OP_CONVOLVE_HPP__

#include <cmath>
#include <boost/serialization/version.hpp>
#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <cuv/libs/nlmeans/conv3d.hpp>
#ifndef NO_THEANO_WRAPPERS
#      include <cuv/convolution_ops/convolution_ops_theano.hpp>
#endif
#include <log4cxx/logger.h>
namespace cuvnet
{
    class monitor; // used in weighted_sub_tensor_op

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
                unsigned int m_nGroups;//number of groups of filters
                unsigned int m_partial_sum;
                int m_padding_start;
                int m_padding_size;
                int m_stride;
                cuv::tensor<int, matrix::memory_space_type> m_indices;
            public:
                inline int stride()const{return m_stride;}
                inline int padding_start()const{return m_padding_start;}
                inline int padding_size()const{return m_padding_size;}
                Convolve() :Op(2,1){} ///< for serialization

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

                void set_random_sparse();

                /// @return true iff we use padding
                inline bool is_padded()const{
                    return m_padding_start != 0;
                }

                /// change the partial_sum parameter
                inline void set_partial_sum(int ps){
                    m_partial_sum = ps;
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
                    }
        };





#ifndef NO_THEANO_WRAPPERS
    /**
     * convolve a set of images using a set of filters.
     *
     * This is the "neural net" type convolution, where filters are 4D and
     * results are summed over input channels.
     *
     * First param, the images, must have shape 
     *
     *  nImages x nChannels x nPixels x nPixels
     *
     * while filters have shape
     *  
     *  nFilt x nFiltChannels x nFiltPiy x nFiltPix 
     *
     *  Bias is optional parameter, if not passed to the constructor, just convolution is performed. 
     *  If bias is passed, then first "full" convolution is done as usual, and to the result of the convolution, the bias is added. 
     *  The bias has non-zero value at the "border" of the result, for all elements of the result which are affected by padding in full convolution.
     *  All other elements are zero, and are not updated during training. 
     *
     *  @ingroup Ops
     *
     */
    class Convolve2dTheano
       :public Op{
           public:
               typedef Op::value_type    value_type;
               typedef Op::op_ptr        op_ptr;
               typedef Op::value_ptr     value_ptr;
               typedef Op::param_t       param_t;
               typedef Op::result_t      result_t;
           private:
           public:
               Convolve2dTheano() :Op(2,1){} ///< for serialization
               std::string m_mode;
               bool m_use_bias;
               value_type m_extended_orig;
               value_type m_extended;
                /**
                 * constructor.
                 *
                 * @param images nImages x nChannels x nPixels x nPixels 
                 * @param filters nFilt x nFiltChannels x nFiltPix x nFiltPix
                 */
               Convolve2dTheano(result_t& images, result_t& filters, std::string mode = "valid")
                   :Op(2,1),
                   m_mode(mode),
                   m_use_bias(false)
               {
                   add_param(0,images);
                   add_param(1,filters);
               }


               Convolve2dTheano(result_t& images, result_t& filters, result_t& bias, std::string mode = "valid")
                   :Op(3,1),
                   m_mode(mode),
                   m_use_bias(true)
               {
                   add_param(0,images);
                   add_param(1,filters);
                   add_param(2,bias);
               }
               void fprop();
               void bprop();
               void _determine_shapes();

           private:
               friend class boost::serialization::access;
               template<class Archive>
                   void serialize(Archive& ar, const unsigned int version){
                       ar & boost::serialization::base_object<Op>(*this);
                       ar & m_mode & m_use_bias & m_extended_orig;
                   }
       };


#endif


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
     * Separable Filter 1d.
     *
     * @ingroup Ops
     */
    class SeparableFilter1d
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                //matrix m_kernel;
                cuv::tensor<float,cuv::host_memory_space> m_kernel;
                cuv::tensor<float,cuv::host_memory_space> m_kernel_reverse;
                unsigned int m_dim;
            public:
                SeparableFilter1d() :Op(1,1){} ///< for serialization.
                /**
                 * ctor.
                 * @param images the input images
                 * @param kernel a kernel used for row and column convolutions.
                 */
                //SeparableFilter1d(result_t& images, const matrix& kernel)
                SeparableFilter1d(result_t& images, const cuv::tensor<float,cuv::host_memory_space>& kernel, unsigned int dim)
                    :Op(1,1),
                    m_kernel(kernel),
                    m_dim(dim)
                {
                    assert(dim < 3);
                    add_param(0,images);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

                inline
                void set_kernel(const matrix& kernel){ m_kernel = kernel; }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_kernel;
                        ar & m_dim;
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
                matrix m_kernel_reverse;
            public:
                SeparableFilter() :Op(1,1){} ///< for serialization.
                /**
                 * ctor.
                 * @param images the input images
                 * @param kernel a kernel used for row and column convolutions.
                 */
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
     * Response Norm across maps.
     *
     * @ingroup Ops
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
                LocalPooling() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 * @subx pooling size
                 * @stridex distance between neighboring neurons in a bank
                 * @param pt pooling type
                 */
                LocalPooling(result_t& images, int subsx, int stridex, cuv::alex_conv::pool_type pt)
                    :Op(1,1),
                    m_pooltype(pt),
                    m_subsx(subsx),
                    m_stridex(stridex)
                {
                    add_param(0,images);
                }
                inline unsigned int subsx()const{return m_subsx;}
                inline unsigned int stridex()const{return m_stridex;}
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

    /**
     * Calculate the norm  or max out of consecutive elements in the input.
     *
     * Expressed in numpy style, this calculates:
     * 
     * f(X) = sqrt(X[::2, ...] ** 2 + X[1::2, ...] ** 2 + epsilon)
     *
     * @ingroup Ops
     */
    class Tuplewise_op
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
                unsigned int m_dim;
                unsigned int m_subspace_size;
                float m_epsilon;
                cuv::alex_conv::tuplewise_op_functor m_to;
            private:
            public:
                Tuplewise_op() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param images the input images
                 */
                Tuplewise_op(result_t& images, unsigned int dim, unsigned int subspace_size, cuv::alex_conv::tuplewise_op_functor to, float epsilon)
                    :Op(1,1),
                    m_dim(dim),
                    m_subspace_size(subspace_size),
                    m_epsilon(epsilon),
                    m_to(to)
                {
                    add_param(0,images);
                }
                void fprop();
                void bprop();

                void _determine_shapes();
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this) & m_dim & m_subspace_size & m_to & m_epsilon;
                    }
        };

        /**
         * Calculates op out of consecutive elements in the input.
         * The input must be of shape n_maps x img_data x n_batch or of shape n_maps x img_data_X x img_data_Y x n_batch
         *
         * Expressed in numpy style, this calculates:
         *
         * f(X) = op((W[::2]* X[::2, ...], W[1::2]* X[1::2, ...], .. )
         *
         * @ingroup Ops
         */
        class Weighted_Sub_Tensor_op
            : public Op{
                public:
                    typedef Op::value_type    value_type;
                    typedef Op::op_ptr        op_ptr;
                    typedef Op::value_ptr     value_ptr;
                    typedef Op::param_t       param_t;
                    typedef Op::result_t      result_t;

                    // for profiling host / dev
                    typedef cuv::tensor<unsigned char,cuv::dev_memory_space> char_matrix;
                    unsigned int m_size;
                    unsigned int m_subspace_size;
                    unsigned int m_stride;
                    cow_ptr<char_matrix>  m_max_idx;
                    value_ptr    m_lae;
                    boost::shared_ptr<cuvnet::monitor> m_S;     
                    cuv::alex_conv::weighted_sub_tensor_op_functor m_to;
                    float m_eps;
                    bool m_spn;
                    bool m_memory_flag;
                private:
                public:
                    Weighted_Sub_Tensor_op() :Op(2,1){} ///< for serialization
                    /**
                     * ctor.
                     * @param images the input images
                     */
                    Weighted_Sub_Tensor_op(result_t& images, result_t& m_W, boost::shared_ptr<cuvnet::monitor> S, unsigned int size, unsigned int stride, unsigned int subspace_size, cuv::alex_conv::weighted_sub_tensor_op_functor to, float eps, bool spn)
                        :Op(2,1),
                        m_size(size),
                        m_subspace_size(subspace_size),
                        m_stride(stride),
                        m_to(to),
                        m_eps(eps),
                        m_spn(spn)
                    {
                        add_param(0,images);
                        add_param(1,m_W);

                        using namespace cuv;
                           //generate dummy tensor ( empty ) to avoid null pointer exceptions
                        value_ptr z(new value_type(0, value_ptr::s_allocator));
                        m_lae = z;
                        
                        m_S = S;
                        m_memory_flag = false;
                        //generate dummy tensor ( empty ) to avoid null pointer exceptions
                        cow_ptr<char_matrix> m(new char_matrix(0, value_ptr::s_allocator));
                        m_max_idx = m;                          
                    }

                    void fprop();
                    void bprop();

                    void _determine_shapes();
                    virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                    void set_S(boost::shared_ptr<cuvnet::monitor> S){
                        m_S = S;
                    }
                                       
                private:
                    friend class boost::serialization::access;
                    template<class Archive>
                        void serialize(Archive& ar, const unsigned int version){
                            ar & boost::serialization::base_object<Op>(*this) & m_subspace_size & m_size & m_to & m_stride & m_eps & m_spn;
                        }
            };
////////////////////////////////// fully connected SPN ///////////////////////////////////////

        /**
         * TODO
	 * @ingroup Ops
         */
        class Spn_Sum_Fully_Connected
            : public Op{
                public:
                    typedef Op::value_type    value_type;
                    typedef Op::op_ptr        op_ptr;
                    typedef Op::value_ptr     value_ptr;
                    typedef Op::param_t       param_t;
                    typedef Op::result_t      result_t;

                    // for profiling host / dev
                    value_ptr    m_lae;
		    unsigned int m_size;
		    std::vector<unsigned int> src_shape;
		    std::vector<unsigned int> src_reshape;
		    bool m_reshape;
                private:
                public:
                    Spn_Sum_Fully_Connected() :Op(2,1){} ///< for serialization
                    /**
                     * ctor.
                     * @param images the input images
                     */
                    Spn_Sum_Fully_Connected(result_t& images, result_t& m_W, unsigned int size, bool reshape = false)
                        :Op(2,1),
			m_size(size),
			m_reshape(reshape)
                    {
                        add_param(0,images);
                        add_param(1,m_W);

                        using namespace cuv;
                           //generate dummy tensor ( empty ) to avoid null pointer exceptions
                        value_ptr z(new value_type(0, value_ptr::s_allocator));
                        m_lae = z;
                    }

                    void fprop();
                    void bprop();

                    void _determine_shapes();
                    virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                                       
                private:
                    friend class boost::serialization::access;
                    template<class Archive>
                        void serialize(Archive& ar, const unsigned int version){
                            ar & boost::serialization::base_object<Op>(*this) & m_size & m_reshape;
                        }
            };
}
BOOST_CLASS_VERSION(cuvnet::Convolve, 1)
#endif /* __OP_CONVOLVE_HPP__ */
