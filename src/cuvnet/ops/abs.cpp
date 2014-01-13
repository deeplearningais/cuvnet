#include "abs.hpp"

namespace cuvnet
{
    void Abs::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();

        if(r0.can_overwrite_directly()){
            apply_scalar_functor( r0.overwrite_or_add_value().data(), inp, SF_ROBUST_ABS, m_scalar);
        }else{
            value_ptr res(new value_type(inp.shape(), value_ptr::s_allocator));
            apply_scalar_functor( *res, inp, SF_ROBUST_ABS, m_scalar);

            r0.push(res); // 'copy' a newly created matrix
        }

        if(!p0.need_derivative)
            p0.value.reset();
    }

    void Abs::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            value_type& res = p0.overwrite_or_add_value();
            apply_scalar_functor(res,p0.value.cdata(),SF_DROBUST_ABS, m_scalar);
            res *= r0.delta.cdata();
        }else{
            value_ptr res(new value_type(p0.value.cdata().shape(), value_ptr::s_allocator));
            apply_scalar_functor(*res,p0.value.cdata(),SF_DROBUST_ABS, m_scalar);
            *res *= r0.delta.cdata(); 
            p0.push(res);
        }
        r0.delta.reset();
        p0.value.reset(); // now we don't need it anymore ;)
    }
}
