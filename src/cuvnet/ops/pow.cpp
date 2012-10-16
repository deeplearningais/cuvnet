#include "pow.hpp"

namespace cuvnet
{
    
    void Pow::_graphviz_node_desc(detail::graphviz_node& desc)const{
        if(m_exponent == 0.5f)
            desc.label = "sqrt (x)";
        else if(m_exponent == 2.f)
            desc.label = "x^2";
        else
            desc.label = "pow (x," + boost::lexical_cast<std::string>(m_exponent) + ")";
    }

    void Pow::fprop(){
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

    void Pow::bprop(){
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
        p0.value.reset();
    }
}
