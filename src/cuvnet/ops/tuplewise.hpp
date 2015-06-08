#ifndef __OP_TUPLEWISE_HPP__
#     define __OP_TUPLEWISE_HPP__

#include <cmath>
#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <log4cxx/logger.h>
namespace cuvnet
{
    class monitor; // used in WeightedSubtensor

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

}
#endif /* __OP_TUPLEWISE_HPP__ */

