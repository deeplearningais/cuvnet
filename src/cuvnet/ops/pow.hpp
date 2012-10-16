#ifndef __OP_POW_HPP__
#     define __OP_POW_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Raises its input to the \f$n\f$-th power (elementwise).
     *
     * @ingroup Ops
     */
    class Pow
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                float m_exponent;
            public:
                Pow(){} /// for serialization
                Pow(float exponent, result_t& p0):Op(1,1), m_exponent(exponent){
                    add_param(0,p0);
                }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_exponent;
                    }
        };
}
#endif /* __OP_POW_HPP__ */
