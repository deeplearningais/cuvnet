#include <cuvnet/tools/monitor.hpp>
#include "spn_output.hpp"

namespace cuvnet{


    void Spn_Output_Op::_determine_shapes(){
        cuvAssert(m_params[0]->shape.size() > 0);
    
        //check shape of weight tensor
        cuvAssert(m_params[1]->shape[0] == m_classes);
    
        std::vector<unsigned int> dst = m_params[0]->shape;
        dst[0] = 1;
        m_results[0]->shape = dst;
    }

    
    void Spn_Output_Op::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        param_t::element_type&  p2 = *m_params[2];
        result_t::element_type& r0 = *m_results[0];

            if(r0.can_overwrite_directly()){
                cuv::alex_conv::spn_output_op(*r0.overwrite_or_add_value(),  p0.value.cdata(), p1.value.cdata(), p2.value.cdata());
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                cuv::alex_conv::spn_output_op(*v,                           p0.value.cdata(), p1.value.cdata(), p2.value.cdata());
                m_lae = v;
                r0.push(v);
            }
    }
    
    
   inline  Op::value_ptr  Spn_Output_Op::get_data_ptr(bool can_overwrite, param_t::element_type* p){
        if (can_overwrite) 
            return p->overwrite_or_add_value();
        else{
            Op::value_ptr ptr(new value_type(p->shape, value_ptr::s_allocator));
            return ptr; 
        }
    }

  void Spn_Output_Op::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        param_t::element_type&  p2 = *m_params[2];
        result_t::element_type& r0 = *m_results[0];
        
        //check that S has been set if op is used for spn with soft gradient
        cuvAssert( ((*m_S)["S"].size() == p0.shape[2]) || ((*m_S)["S"].size() == p0.shape[3]) );

            if (p0.need_derivative || p1.need_derivative || p2.need_derivative){
                bool p0_old = p0.can_overwrite_directly();
                bool p1_old = p1.can_overwrite_directly();
                bool p2_old = p2.can_overwrite_directly();
                value_ptr p0_ptr = get_data_ptr(p0_old, &p0);
                value_ptr p1_ptr = get_data_ptr(p1_old, &p1);
                value_ptr p2_ptr = get_data_ptr(p2_old, &p2);
                
                spn_output_op_grad(*p0_ptr, p0.value.cdata(), *p1_ptr, *p2_ptr, p1.value.cdata(), p2.value.cdata(),  (*m_S)["S"], m_lae.cdata(), r0.delta.cdata(), p0.need_derivative, p1.need_derivative, p2.need_derivative, m_eps);       

                if (!p0_old) p0.push(p0_ptr);
                if (!p1_old) p1.push(p1_ptr);                   
                if (!p2_old) p2.push(p2_ptr);
            }
            p0.value.reset();
            p1.value.reset();
            p2.value.reset();
            r0.delta.reset();
    }
    
    
}
