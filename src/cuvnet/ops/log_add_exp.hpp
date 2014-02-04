#ifndef __OP_LOG_ADD_EXP_HPP__
#     define __OP_LOG_ADD_EXP_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * calculates the function log(a + exp(b)) in a stable way ( elementwise ).
     *
     * \f(a,b) = \log_add_exp(a + b )$
     *
     * @ingroup Ops
     */
    class LogAddExp
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                float m_scalar; ///< the pre-factor 
            public:
                LogAddExp(){} ///< for serialization
               //TODO overload function for log_add_exp(p0, p1)
               
                /**
                 *  (Calculates log (a + exp(a))
                 * ctor.
                 * @param factor scalar factor
                 * @param p0 the input a
                 * */
                LogAddExp(float factor, result_t& p0):Op(1,1), m_scalar(factor) {
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
#endif /* __OP_LOG_ADD_EXP_HPP__ */
