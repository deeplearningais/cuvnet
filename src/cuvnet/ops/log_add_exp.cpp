#include "log_add_exp.hpp"

namespace cuvnet
{
    void LogAddExp::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();
        value_ptr res(new value_type(inp.shape(), value_ptr::s_allocator));

        apply_scalar_functor( *res, inp, SF_LOGADDEXP, m_scalar);

        r0.push(res); // 'copy' a newly created matrix
    }

    void LogAddExp::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);
        
        //e^x / (exp(a) + e^x)
        if(p0.can_overwrite_directly()){
            value_type v = *p0.overwrite_or_add_value();
            apply_scalar_functor(v, p0.value.cdata(), SF_LOGADDEXP_GRAD, m_scalar);
            v *= r0.delta.cdata();
        } else {
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            apply_scalar_functor(*v, p0.value.cdata(), SF_LOGADDEXP_GRAD, m_scalar);
            *v *= r0.delta.cdata(); 
            p0.push(v);
        }
        r0.delta.reset();
        p0.value.reset();        
    }
}
