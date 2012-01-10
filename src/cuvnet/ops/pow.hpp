#ifndef __OP_POW_HPP__
#     define __OP_POW_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
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

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();
                    value_ptr res(new value_type(inp.shape()));

                    apply_scalar_functor( *res,
                            inp, SF_POW, m_exponent);

                    r0.push(res); // 'copy' a newly created matrix

                    if(!p0.need_derivative)
                        p0.value.reset();       // forget it
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    const value_type& inp = p0.value.cdata();
                    value_ptr res(new value_type(inp.shape()));
                    apply_scalar_functor(*res,inp,SF_DPOW, m_exponent);
                    *res *= r0.delta.cdata(); // TODO: write BF_POW_TIMES functor in cuv
                    r0.delta.reset();
                    p0.push(res);
                }
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
