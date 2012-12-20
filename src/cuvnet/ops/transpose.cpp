#include "transpose.hpp"

namespace cuvnet
{
    void Transpose::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();           // original

        if(r0.can_overwrite_directly()){
            cuv::transpose(*r0.overwrite_or_add_value(), inp);
        }else{
            value_ptr p(new value_type(r0.shape));
            cuv::transpose(*p, inp);
            r0.push(p);
        }
        
    }
    void Transpose::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = r0.delta.cdata();           // original

        if(p0.can_overwrite_directly()){
            cuv::transpose(*p0.overwrite_or_add_value(), inp);
        }else{
            value_ptr p(new value_type(p0.shape));
            cuv::transpose(*p, inp);
            p0.push(p);
        }
    }
    void Transpose::_determine_shapes(){
        cuvAssert(m_params[0]->shape.size() == 2);
        
        std::vector<unsigned int> shape(2);
        shape[0] = m_params[0]->shape[1];
        shape[1] = m_params[0]->shape[0];
        m_results[0]->shape = shape;
    }
}
