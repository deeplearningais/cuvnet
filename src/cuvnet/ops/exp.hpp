#ifndef __OP_EXP_HPP__
#     define __OP_EXP_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
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

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();
                    value_ptr res(new value_type(inp.shape()));

                    if(m_scalar != 1.f)
                        apply_scalar_functor( *res, inp, SF_MULT, m_scalar);
                    apply_scalar_functor( *res, SF_EXP);

                    r0.push(res); // 'copy' a newly created matrix

                    p0.value.reset();       // forget params
                    
                    if(p0.need_derivative)
                        m_res = res; // do not forget result, we need it for bprop!
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    apply_scalar_functor(*m_res,*m_res,SF_MULT, m_scalar);
                    *m_res *= r0.delta.cdata(); 
                    r0.delta.reset();
                    p0.push(m_res);
                    m_res.reset();
                }
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
