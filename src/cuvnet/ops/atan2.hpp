#ifndef __OP_ATAN2_HPP__
#     define __OP_ATAN2_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * calculates pointwise atan2(Y, X), where X, Y are tensors
     *
     * @ingroup Ops
     */
    class Atan2
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            public:
                Atan2(){} /// for serialization
                /**
                 * @param p0 y
                 * @param p1 x
                 */
                Atan2(result_t& p0, result_t& p1)
                    :Op(2,1){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                void fprop();
                void bprop();
                void _determine_shapes(){
                    assert(m_params[0]->shape == m_params[1]->shape);
                    m_results[0]->shape = m_params[0]->shape;
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}

#endif /* __OP_ATAN2_HPP__ */
