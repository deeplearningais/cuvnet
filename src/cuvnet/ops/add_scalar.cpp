#include "add_scalar.hpp"
#include <boost/format.hpp>

namespace cuvnet
{
    /***************************************************
     *  AddScalar
     ***************************************************/
    void AddScalar::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        // TODO cuv: a = a + b*scalar
        if(r0.can_overwrite_directly()){
            apply_scalar_functor(r0.overwrite_or_add_value().data(),p0.value.cdata(),SF_ADD,m_scalar);
        }
        else if(r0.can_add_directly()){
            r0.overwrite_or_add_value().data()+=p0.value.cdata();
            r0.overwrite_or_add_value().data()+=m_scalar;
        }else{
            // reallocate *sigh*
            value_ptr v = p0.value;
            p0.value.reset(); // try to overwrite p0
            *v += m_scalar;
            r0.push(v);
        }
        p0.value.reset();
    }

    void AddScalar::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        assert(p0.need_derivative);
        p0.push(r0.delta);
        r0.delta.reset();
    }
    void AddScalar::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = boost::str(boost::format("x + %2.3f")%m_scalar);
    }

    /***************************************************
     *  SubtractFromScalar
     ***************************************************/
    void SubtractFromScalar::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = boost::str(boost::format("%2.3f - x")%m_scalar);
    }

    void SubtractFromScalar::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        // TODO cuv: a = a + b*scalar SAXPY
        if(r0.can_overwrite_directly()){
            apply_scalar_functor(r0.overwrite_or_add_value().data(),p0.value.cdata(),SF_RSUB,m_scalar);
        }
        else if(r0.can_add_directly()){
            r0.overwrite_or_add_value().data()+=m_scalar;
            r0.overwrite_or_add_value().data()-=p0.value.cdata();
        }else{
            // reallocate *sigh*
            value_ptr v = p0.value;
            p0.value.reset(); // try to overwrite p0
            apply_scalar_functor(*v,SF_NEGATE); // SAXPY!
            *v += m_scalar;
            r0.push(v);
        }
        p0.value.reset();
    }

    void SubtractFromScalar::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        assert(p0.need_derivative);
        if(p0.can_overwrite_directly()){
            apply_scalar_functor(p0.overwrite_or_add_value().data(),r0.delta.cdata(),SF_NEGATE);
        }else if(p0.can_add_directly()){
            apply_scalar_functor(r0.delta.data(),SF_NEGATE);
            p0.overwrite_or_add_value().data()+=r0.delta.cdata(); // SAXPY!
        }else{
            apply_scalar_functor(r0.delta.data(),SF_NEGATE);
            p0.push(r0.delta);
        }
        r0.delta.reset();
    }
}
