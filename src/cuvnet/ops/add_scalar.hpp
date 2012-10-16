#ifndef __OP_ADD_SCALAR_HPP__
#     define __OP_ADD_SCALAR_HPP__

#include <cuvnet/op.hpp>
namespace cuvnet
{
    /**
     * Adds a scalar to its inputs.
     * @ingroup Ops
     */
    class AddScalar
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                float m_scalar;
            public:
                /// default ctor for serialization only.
                AddScalar()
                    : Op(1,1), m_scalar(0)
            {
            }
                AddScalar(result_t& mat, float f)
                    :   Op(1,1), m_scalar(f)
            {
                add_param(0,mat);
            }
                AddScalar(float f, result_t& mat)
                    :   Op(1,1), m_scalar(f)
            {
                add_param(0,mat);
            }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
    };

    /**
     * Subtracts inputs from a scalar.
     *
     * @ingroup Ops
     */
    class SubtractFromScalar
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                float m_scalar;
            public:
                SubtractFromScalar() :   Op(1,1){}
                SubtractFromScalar(result_t& mat, float f)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }
                SubtractFromScalar(float f, result_t& mat)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
    };
}
#endif /* __OP_ADD_SCALAR_HPP__ */
