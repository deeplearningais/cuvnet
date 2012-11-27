#ifndef __OP_MAT_PLUS_VEC_HPP__
#     define __OP_MAT_PLUS_VEC_HPP__

#include <cuvnet/op.hpp>
namespace cuvnet
{
    /**
     * adds a vector to a tensor.
     *
     * Depending on the axis parameter, the vector must have the same length as
     * the first or last dimension of the tensor.
     *
     * @ingroup Ops
     */
    class MatPlusVec
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                unsigned int m_axis;
            public:
                MatPlusVec():Op(2,1){} ///< for serialization
                
                /**
                 * ctor.
                 *
                 * @param mat the matrix 
                 * @param vec the vector
                 * @param axis the axis-th dimension of matrix must agree with the vector dimension.
                 *
                 * @warning currently, only the first and the last dimension of matrix are supported.
                 */
                MatPlusVec(result_t& mat, result_t& vec, unsigned int axis)
                    :   Op(2,1)
                      , m_axis(axis)
            {
                add_param(0,mat);
                add_param(1,vec);
            }

                void fprop();
                void bprop();
                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };

    /**
     * Multiply a matrix with a vector.
     *
     * Depending on the axis parameter, the vector must have the same length as
     * the first or last dimension of the tensor.
     *
     * @ingroup Ops
     */
    class MatTimesVec
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                unsigned int m_axis;
            public:
                MatTimesVec():Op(2,1){} ///< for serialization
                /**
                 * ctor.
                 * @param mat the n-dimensional matrix
                 * @param vec the vector
                 * @param axis the axis-th dimension of matrix must agree with the vector dimension.
                 * @note axis can currently only be 0 or n-1.
                 */
                MatTimesVec(result_t& mat, result_t& vec, unsigned int axis)
                    :   Op(2,1)
                      , m_axis(axis)
            {
                add_param(0,mat);
                add_param(1,vec);
            }

                void fprop();
                void bprop();
                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };
}
#endif /* __OP_MAT_PLUS_VEC_HPP__ */
