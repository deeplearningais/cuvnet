#ifndef __OP_TRANSPOSE_HPP__
#     define __OP_TRANSPOSE_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Transpose of a 2-dimensional matrix.
     *
     * \ingroup Ops
     */
    class Transpose
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Transpose(){} ///< for serialization

                /**
                 * ctor.
                 * @param p0 the input, which is to be transposed
                 */
                Transpose(result_t& p0):Op(1,1){
                    add_param(0,p0);
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
#endif /* __OP_REPMAT_HPP__ */
