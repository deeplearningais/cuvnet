#include "exp.hpp"

namespace cuvnet
{
    void Exp::fprop(){
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

    void Exp::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(m_scalar != 1.f)
            apply_scalar_functor(*m_res,*m_res,SF_MULT, m_scalar);
        *m_res *= r0.delta.cdata(); 
        r0.delta.reset();
        p0.push(m_res);
        m_res.reset();
    }
}
