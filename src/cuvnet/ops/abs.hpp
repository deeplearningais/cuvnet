#ifndef __OP_ABS_HPP__
#     define __OP_ABS_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    class Abs
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                float m_scalar;
            public:
                Abs(){} /// for serialization
                Abs(result_t& p0, float epsilon=0.0001f):Op(1,1), m_scalar(epsilon){
                    add_param(0,p0);
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();

                    if(r0.can_overwrite_directly()){
                        apply_scalar_functor( r0.overwrite_or_add_value().data(), inp, SF_ROBUST_ABS, m_scalar);
                    }else{
                        value_ptr res(new value_type(inp.shape()));
                        apply_scalar_functor( *res, SF_ROBUST_ABS, m_scalar);

                        r0.push(res); // 'copy' a newly created matrix
                    }

                    if(!p0.need_derivative)
                        p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    if(p0.can_overwrite_directly()){
                        value_type& res = p0.overwrite_or_add_value();
                        apply_scalar_functor(res,p0.value.cdata(),SF_DROBUST_ABS, m_scalar);
                        res *= r0.delta.cdata();
                    }else{
                        value_ptr res(new value_type(p0.value.cdata().shape()));
                        apply_scalar_functor(*res,p0.value.cdata(),SF_DROBUST_ABS, m_scalar);
                        *res *= r0.delta.cdata(); 
                        p0.push(res);
                    }
                    r0.delta.reset();
                    p0.value.reset(); // now we don't need it anymore ;)
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
#endif /* __OP_ABS_HPP__ */
