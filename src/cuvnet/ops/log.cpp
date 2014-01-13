#include "log.hpp"

namespace cuvnet
{
    void Log::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();
        value_ptr res(new value_type(inp.shape(), value_ptr::s_allocator));

        apply_scalar_functor( *res,
                inp, SF_LOG);

        r0.push(res); // 'copy' a newly created matrix

        if(!p0.need_derivative)
            p0.value.reset();       // forget it
    }
    void Log::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        // try overwriting p0
        const value_type& inp = p0.value.cdata();
        value_type&       out = p0.value.data_onlyshape();
        apply_scalar_functor(out,inp,SF_INV);
        out *= r0.delta.cdata();
        r0.delta.reset();
        p0.push(p0.value);
    }
}
