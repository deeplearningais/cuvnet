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
                bool m_subtract_mean;
            public:
                MatPlusVec():Op(2,1),m_subtract_mean(false){} ///< for serialization
                
                /**
                 * ctor.
                 *
                 * @param mat the matrix 
                 * @param vec the vector
                 * @param axis the axis-th dimension of matrix must agree with the vector dimension.
                 * @param subtract_mean if true, ensure that the matrix gradient is zero-mean
                 *
                 * @warning currently, only the first and the last dimension of matrix are supported.
                 */
                MatPlusVec(result_t& mat, result_t& vec, unsigned int axis, bool subtract_mean=false)
                    :   Op(2,1)
                      , m_axis(axis)
                      , m_subtract_mean(subtract_mean)
            {
                add_param(0,mat);
                add_param(1,vec);
            }

                void fprop();
                void bprop();
                void _determine_shapes();
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                        if(version > 0)
                            ar & m_subtract_mean;
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
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };

    /**
     * Divide a matrix with a vector.
     *
     * Depending on the axis parameter, the vector must have the same length as
     * the first or last dimension of the tensor.
     *
     * @ingroup Ops
     */
    class MatDivideVec
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
                MatDivideVec():Op(2,1){} ///< for serialization
                /**
                 * ctor.
                 * @param mat the n-dimensional matrix
                 * @param vec the vector
                 * @param axis the axis-th dimension of matrix must agree with the vector dimension.
                 * @note axis can currently only be 0 or n-1.
                 */
                MatDivideVec(result_t& mat, result_t& vec, unsigned int axis)
                    :   Op(2,1)
                      , m_axis(axis)
            {
                add_param(0,mat);
                add_param(1,vec);
            }

                void fprop();
                void bprop();
                void _determine_shapes();
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };
}
BOOST_CLASS_VERSION(cuvnet::MatPlusVec, 1);
#endif /* __OP_MAT_PLUS_VEC_HPP__ */
