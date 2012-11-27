#ifndef __OP_EXP_HPP__
#     define __OP_EXP_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * calculates the exponential function of its inputs (elementwise), possibly with a scalar pre-factor.
     *
     * \f$y = \exp(ax)\f$
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

                float m_scalar; ///< the pre-factor 
                value_ptr m_res; ///< stores the result, which is needed for backpropagation
            public:
                Exp(){} ///< for serialization
                /**
                 * ctor.
                 * @param factor input is multiplied by this before exp
                 * @param p0 the input
                 */
                Exp(float factor, result_t& p0):Op(1,1), m_scalar(factor){
                    add_param(0,p0);
                }

                void fprop();
                void bprop();
                void release_data();
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
