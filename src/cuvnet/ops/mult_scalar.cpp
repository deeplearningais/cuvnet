#include "mult_scalar.hpp"
#include <boost/format.hpp>

namespace cuvnet
{
    void MultScalar::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = boost::str(boost::format("%2.3f x")%m_scalar);
    }

    void MultScalar::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            apply_scalar_functor(r0.overwrite_or_add_value().data(),p0.value.cdata(),SF_MULT,m_scalar);
        }
        else if(r0.can_add_directly()){
            apply_binary_functor(r0.overwrite_or_add_value().data(),p0.value.cdata(),BF_XPBY,m_scalar);
        }else{
            // reallocate *sigh*
            value_ptr v = p0.value;
            p0.value.reset(); // try to overwrite p0
            *v *= m_scalar;
            r0.push(v);
        }
        p0.value.reset();
    }

    void MultScalar::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        assert(p0.need_derivative);
        if(p0.can_overwrite_directly()){
            apply_scalar_functor(p0.overwrite_or_add_value().data(), r0.delta.cdata(), SF_MULT, m_scalar);
        }else if(p0.can_add_directly()){
            apply_binary_functor(p0.overwrite_or_add_value().data(), r0.delta.cdata(), BF_XPBY, m_scalar);
        }else{
            value_ptr v = r0.delta; // try to overwrite r0.delta
            r0.delta.reset();
            *v *= m_scalar;
            p0.push(v);
        }
        r0.delta.reset();
    }
}
