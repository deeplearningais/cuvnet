#ifndef __OP_ABS_HPP__
#     define __OP_ABS_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Calculates the absolute values of its inputs.
     *
     * More precisely, to be able to take the derivative, we calculate
     *
     * \f[y = sqrt( x^2 + \epsilon )\f]
     *
     * \ingroup Ops
     */
    class Abs
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                float m_scalar; ///< a small value which is added before taking the square root
            public:
                Abs(){} ///< for serialization

                /**
                 * ctor.
                 * @param p0 the value of which to take the absolute
                 * @param epsilon added before taking square root of x^2
                 */
                Abs(result_t& p0, float epsilon=0.0001f):Op(1,1), m_scalar(epsilon){
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
#endif /* __OP_ABS_HPP__ */
