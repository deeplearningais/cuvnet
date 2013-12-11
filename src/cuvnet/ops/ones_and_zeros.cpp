#include "ones_and_zeros.hpp"

namespace cuvnet
{
    void ScalarLike::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        if(r0.can_overwrite_directly()){
            value_ptr& res = r0.overwrite_or_add_value(); // Note: /must/ be reference, otherwise copied in next step!
            *res = m_scalar;
        }else if(r0.can_add_directly()){
            value_ptr& res = r0.overwrite_or_add_value(); // Note: /must/ be reference, otherwise copied in next step!
            *res += m_scalar;
        }else{
            // try to overwrite inputs: we don't need them for bprop.
            *p0.value = m_scalar;
            r0.push(p0.value); // 'copy' a newly created matrix
        }
        p0.value.reset();
    }

    void ScalarLike::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        *r0.delta = 0.f;
        p0.push(r0.delta);

        //m_result.dealloc();
        r0.delta.reset();
    }
}
