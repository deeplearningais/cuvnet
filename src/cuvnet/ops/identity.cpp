#include "identity.hpp"

namespace cuvnet
{
    void Identity::fprop(){
        // identity
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        if(r0.can_add_directly()){
            value_ptr& ptr = r0.overwrite_or_add_value();
            *ptr          += p0.value.cdata();
        }else{
            r0.push(p0.value); // 'copy' a newly created matrix
        }
        p0.value.reset();       // don't need that for backprop etc.
    }

    void Identity::bprop(){
        // identity
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        if(p0.can_add_directly()){
            value_ptr& ptr = p0.overwrite_or_add_value();
            *ptr          += r0.delta.cdata();
        }else{
            r0.push(p0.value); // 'copy' a newly created matrix
        }
        r0.delta.reset();
    }
}
