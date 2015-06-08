#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/ops/weighted_sub_tensor_op.hpp>

namespace cuvnet{

/************************************************************************
     * Weighted_SubTensor_op
     * ***********************************************************************/

    void WeightedSubtensor::_graphviz_node_desc(detail::graphviz_node& desc)const{
        using namespace cuv;
        using namespace alex_conv;
        switch(m_to){
            case WST_WMAX:
                desc.label = "WST_WMAX";
                break;
            case WST_LOGWADDEXP:
                desc.label = "WST_LOGWADDEXP";
                break;
            case WST_LOGWADDEXP_LOGSPACE:
                desc.label = "WST_LOGWADDEXP_LOGSPACE";
                break;
            case WST_WMAX_LOGSPACE:
                desc.label = "WST_WMAX_LOGSPACE";
                break;
            case WST_WADD:
                desc.label = "WST_WADD";
                break;                
        }
    }

    void WeightedSubtensor::_determine_shapes(){
        cuvAssert(m_params[0]->shape.size() > 1);
        cuvAssert(m_params[0]->shape.size() > 0);
    
    //check shape of weight tensor
    cuvAssert(m_params[1]->shape[0] == m_size);
        cuvAssert(m_params[1]->shape[1] == m_subspace_size);
    
        std::vector<unsigned int> dst = m_params[0]->shape;
        dst[0] = m_size;
        m_results[0]->shape = dst;
    }

    void WeightedSubtensor::fprop(){      
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
        if ((m_to == cuv::alex_conv::WST_WMAX) || (m_to == cuv::alex_conv::WST_WMAX_LOGSPACE)){
                if (!m_memory_flag){
                    cow_ptr< char_matrix > m(new char_matrix(r0.shape, value_ptr::s_allocator));
                    m_max_idx  = m;
                    m_memory_flag = true;  
                 }

            if(r0.can_overwrite_directly()){
                value_ptr& v = r0.overwrite_or_add_value();
                weighted_subtensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                weighted_subtensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
                r0.push(v);
            }
        }else{
            if(r0.can_overwrite_directly()){
                value_ptr& v = r0.overwrite_or_add_value();
                weighted_subtensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
                m_lae = v;
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                weighted_subtensor_op(*v, *m_max_idx, p0.value.cdata(), p1.value.cdata(), m_size, m_stride, m_subspace_size, m_to, m_eps);
                m_lae = v;
                r0.push(v);
            }
        }
   
    }

    void WeightedSubtensor::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p1 = *m_params[0];
        param_t::element_type&  p2 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        
        if (p1.need_derivative || p2.need_derivative){
            if(p1.can_overwrite_directly()){
                if (p2.can_overwrite_directly()){
                    weighted_subtensor_op_grad(*p1.overwrite_or_add_value(), *p2.overwrite_or_add_value(), p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(),  value_type(), m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
                }else{
                    value_ptr ptr(new value_type(p2.shape, value_ptr::s_allocator));
                    weighted_subtensor_op_grad(*p1.overwrite_or_add_value(), *ptr, p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(), value_type(), m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
                    p2.push(ptr);
                }
            }else{
                value_ptr ptr(new value_type(p1.shape, value_ptr::s_allocator));
                if (p2.can_overwrite_directly()){
                    weighted_subtensor_op_grad(*ptr, *p2.overwrite_or_add_value(), p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(),  value_type(), m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
                } else{
                    value_ptr ptr2(new value_type(p2.shape, value_ptr::s_allocator));
                    weighted_subtensor_op_grad(*ptr, *ptr2, p1.value.cdata(), r0.delta.cdata(), p2.value.cdata(), m_lae.cdata(),  value_type(), m_max_idx.cdata(), m_spn, p1.need_derivative, p2.need_derivative, m_size, m_stride, m_subspace_size, m_to, m_eps);
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
