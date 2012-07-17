#ifndef __RECTIFIED_LINEAR_HPP__
#     define __RECTIFIED_LINEAR_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Calculates the maximum with 0.
     *
     * i.e. \f$y_i = \max(0, x_i)\f$
     *
     * \ingroup Ops
     */
    class RectifiedLinear
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                cuv::tensor<unsigned char, matrix::memory_space_type> m_result;
            public:
                RectifiedLinear(){} /// for serialization
                RectifiedLinear(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();

                    if(r0.can_overwrite_directly()){
                        value_ptr& res = r0.overwrite_or_add_value(); // Note: /must/ be reference, otherwise copied in next step!
                        apply_scalar_functor( *res, inp, SF_MAX, 0.f);
                        if(p0.need_derivative){
                            m_result.resize(p0.shape);
                            apply_scalar_functor(m_result, *res, SF_LEQ, 0.f); // 1 iff we cut off
                        }
                    }else{
                        // try to overwrite inputs: we don't need them for bprop.
                        apply_scalar_functor( *p0.value, inp, SF_MAX, 0.f);
                        if(p0.need_derivative){
                            m_result.resize(p0.shape);
                            apply_scalar_functor(m_result, p0.value.cdata(), SF_LEQ, 0.f); // 1 iff we cut off
                        }
                        r0.push(p0.value); // 'copy' a newly created matrix
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    // try to overwrite r0.delta
                    apply_scalar_functor(*r0.delta, SF_MULT, 0.f, &m_result); // set to 0 when we cut off
                    p0.push(r0.delta);

                    m_result.dealloc();
                    r0.delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}
#endif /* __RECTIFIED_LINEAR_HPP__ */

