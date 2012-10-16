#ifndef __OP_EXP_HPP__
#     define __OP_EXP_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * calculates the exponential function of its inputs (elementwise).
     *
     * @ingroup Ops
     */
    class Exp
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                float m_scalar;
                value_ptr m_res;
            public:
                Exp(){} /// for serialization
                Exp(float factor, result_t& p0):Op(1,1), m_scalar(factor){
                    add_param(0,p0);
                }

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
#endif /* __OP_EXP_HPP__ */
