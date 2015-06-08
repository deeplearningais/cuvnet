#ifndef __OP_CONVOLVE_HPP__
#     define __OP_CONVOLVE_HPP__

#include <cmath>
#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <cuv/libs/nlmeans/conv3d.hpp>
#include <log4cxx/logger.h>
namespace cuvnet
{

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
}
#endif /* __OP_CONVOLVE_HPP__ */
