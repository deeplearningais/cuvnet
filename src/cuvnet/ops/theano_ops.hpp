#ifndef __OP_THEANO_OPS_HPP__
#     define __OP_THEANO_OPS_HPP__

#ifndef NO_THEANO_WRAPPERS

#include <cuv/libs/theano_ops/theano_ops.hpp>
#      include <cuv/convolution_ops/convolution_ops_theano.hpp>
#include <cuvnet/op.hpp>
namespace cuvnet
{
    /**
     *  Theano operator, fliping the dimensions 
     *
     * @ingroup TheanoOps
     * 
     */
    class FlipDims
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                std::vector<bool> m_pattern;

            public:
                FlipDims() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                template<std::size_t D>
                    FlipDims(result_t& in, const cuv::extent_gen<D>& eg)
                    :Op(1,1),
                    m_pattern(D)
            {
                add_param(0,in);
                for(unsigned int i=0; i<D; i++){
                    m_pattern[i] = eg.ranges_[i].finish();
                }
            }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_pattern;
                    }
        };

    /**
     *  Theano operator, shuffling the dimensions 
     *
     * @ingroup TheanoOps
     * 
     */
    class ShuffleDim 
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                std::vector<int> m_pattern;
                unsigned int m_ndim;

            public:
                ShuffleDim() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                template<std::size_t D>
                    ShuffleDim(result_t& in, const cuv::extent_gen<D>& eg)
                    :Op(1,1),
                    m_pattern(D)
            {
                add_param(0,in);
                for(unsigned int i=0; i<D; i++){
                    m_pattern[i] = eg.ranges_[i].finish();
                }
            }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_pattern;
                    }
        };

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
     *  @ingroup TheanoOps
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

}

#endif
#endif /* __OP_THEANO_OPS_HPP__ */
