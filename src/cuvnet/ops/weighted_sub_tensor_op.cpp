#include <cuvnet/tools/monitor.hpp>
#include "convolve.hpp"

namespace cuvnet{

/************************************************************************
     * Weighted_SubTensor_op
     * ***********************************************************************/

    void Weighted_Sub_Tensor_op::_determine_shapes(){
        cuvAssert(m_params[0]->shape.size() > 1);
        cuvAssert(m_params[0]->shape.size() > 0);
    
    //check shape of weight tensor
    cuvAssert(m_params[1]->shape[0] == m_size);
        cuvAssert(m_params[1]->shape[1] == m_subspace_size);
    
        std::vector<unsigned int> dst = m_params[0]->shape;
        dst[0] = m_size;
        m_results[0]->shape = dst;
    }

    void Weighted_Sub_Tensor_op::fprop(){      
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        
        if (false){
            //check for nans
            if (cuv::has_nan(p0.value.cdata())) throw std::runtime_error("ERROR, NAN in input[0] (X)");
            if (cuv::has_nan(p1.value.cdata())) throw std::runtime_error("ERROR, NAN in input[1] (Weights)");
        }
        
        //save max Index for backprop if max function is used
        if ((m_to == cuv::alex_conv::TO_WMAX) || (m_to == cuv::alex_conv::TO_WMAX_LOGSPACE)){
                cow_ptr< char_matrix > m(new char_matrix(r0.shape));
                m_max_idx  = m;

            if(r0.can_overwrite_directly()){
                value_ptr& v = r0.overwrite_or_add_value();
                weighted_sub_tensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                weighted_sub_tensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
                r0.push(v);
            }
        }else{
            if(r0.can_overwrite_directly()){
                value_ptr& v = r0.overwrite_or_add_value();
                weighted_sub_tensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
                m_lae = v;
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                weighted_sub_tensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
                m_lae = v;
                r0.push(v);
            }
        }
   
    }

    void Weighted_Sub_Tensor_op::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p1 = *m_params[0];
        param_t::element_type&  p2 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        
        //check that S has been set if op is used for spn with soft gradient
        if ( m_spn && ((m_to != cuv::alex_conv::TO_WMAX) || (m_to != cuv::alex_conv::TO_WMAX_LOGSPACE)))
            cuvAssert( ((*m_S)["S"].size() == p1.shape[2]) || ((*m_S)["S"].size() == p1.shape[3]) );

            if (p1.need_derivative || p2.need_derivative){
                if(p1.can_overwrite_directly()){
                  if (p2.can_overwrite_directly()){
                 weighted_sub_tensor_op_grad(*p1.overwrite_or_add_value(), *p2.overwrite_or_add_value(), p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(),  (*m_S)["S"], m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
                  }else{
                          value_ptr ptr(new value_type(p2.shape, value_ptr::s_allocator));
                     weighted_sub_tensor_op_grad(*p1.overwrite_or_add_value(), *ptr, p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(), (*m_S)["S"], m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
                      p2.push(ptr);
                }
                       }else{
                           value_ptr ptr(new value_type(p1.shape, value_ptr::s_allocator));
                           if (p2.can_overwrite_directly()){
                    weighted_sub_tensor_op_grad(*ptr, *p2.overwrite_or_add_value(), p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(),  (*m_S)["S"], m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
                   } else{
                             value_ptr ptr2(new value_type(p2.shape, value_ptr::s_allocator));
                    weighted_sub_tensor_op_grad(*ptr, *ptr2, p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(),  (*m_S)["S"], m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
                    p2.push(ptr2);
                  }
                p1.push(ptr);
                       }
                p1.value.reset();
                p2.value.reset();
                r0.delta.reset();
                }

    }
    
}
