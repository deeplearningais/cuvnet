#include <cuvnet/tools/monitor.hpp>
#include "convolve.hpp"

namespace cuvnet{

/************************************************************************
     * Weighted_SubTensor_op
     * ***********************************************************************/

    void Weighted_Sub_Tensor_op::_graphviz_node_desc(detail::graphviz_node& desc)const{
        using namespace cuv;
        using namespace alex_conv;
        switch(m_to){
            case TO_MAX:
                desc.label = "PT_MAX";
                break;
            case TO_LOGWADDEXP:
                desc.label = "TO_LOGWADDEXP";
                break;
            case TO_LOGWADDEXP_LOGSPACE:
                desc.label = "TO_LOGWADDEXP_LOGSPACE";
                break;
            case TO_WMAX_LOGSPACE:
                desc.label = "TO_WMAX_LOGSPACE";
                break;
            case TO_WADD:
                desc.label = "TO_WADD";
                break;                
        }
    }

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
                if (!m_memory_flag){
                    cow_ptr< char_matrix > m(new char_matrix(r0.shape, value_ptr::s_allocator));
                    m_max_idx  = m;
                    m_memory_flag = true;  
                 }

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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    void Spn_Sum_Fully_Connected::_graphviz_node_desc(detail::graphviz_node& desc)const{
        using namespace cuv;
        using namespace alex_conv;
                desc.label = "TO_LOGWADDEXP FULLY CONNECTED";
        
    }

    void Spn_Sum_Fully_Connected::_determine_shapes(){
        cuvAssert(m_params[0]->shape.size() > 1);
        cuvAssert(m_params[0]->shape.size() > 0);
   
        std::vector<unsigned int> dst = m_params[1]->shape;
        dst[0] = m_size;
	if (!m_reshape){
	    dst[1] = m_params[0]->shape[1];
	} else{
	   unsigned int src_size = m_params[0]->shape.size();
           src_shape.resize(src_size);
	   for ( unsigned int i = 0; i < src_size; i++){
		   src_shape[i] = m_params[0]->shape[i];
	   }
	   src_reshape.resize(2);
	   src_reshape[0] = m_params[1]->shape[1];
           src_reshape[1] = m_params[0]->shape[src_size-1];

	   dst[1] = m_params[0]->shape[src_size -1]; 
	}
        m_results[0]->shape = dst;
    }

    void Spn_Sum_Fully_Connected::fprop(){      
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        
        if (false){
            //check for nans
            if (cuv::has_nan(p0.value.cdata())){ 
		    std::cout << "\n\n\n shape: ";
		    for ( unsigned int i = 0; i < src_reshape.size(); i ++)
			    std::cout << "["<< src_reshape[i] << "]";
		    std::cout << std::endl;
		    throw std::runtime_error("ERROR, NAN in input[0] (X)");
	    }
            if (cuv::has_nan(p1.value.cdata())) throw std::runtime_error("ERROR, NAN in input[1] (Weights)");
        }


            value_type v0 = p0.value.cdata();
            if (m_reshape){	    
	       v0.reshape(src_reshape);
	    }	    
           cuvAssert(! cuv::has_nan(v0));

            if(r0.can_overwrite_directly()){
                value_ptr& v = r0.overwrite_or_add_value();
                spn_prod(*v, p1.value.cdata(), v0);
                m_lae = v;
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                spn_prod(*v, p1.value.cdata(), v0);
                r0.push(v);
                m_lae = v;
            }
    }

    void Spn_Sum_Fully_Connected::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p1 = *m_params[0];
        param_t::element_type&  p2 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
       
	value_type v1 = p1.value.cdata();
	value_ptr ptr(new value_type(p1.shape, value_ptr::s_allocator));
	value_type d = *ptr;

	if ( m_reshape ){
		d.reshape(src_reshape);
		v1.reshape(src_reshape);
	}	    
            if (p1.need_derivative){
                spn_prod_grad(d, p2.value.cdata(), r0.delta.cdata(), v1,  m_lae.cdata());
		p1.push(ptr);
            }
	    if (p2.need_derivative){
		    //calculate partial derivatives of weights 
                    value_ptr ptr2(new value_type(p2.shape, value_ptr::s_allocator));
	            
		    value_type srcT(cuv::extents[v1.shape(1)][v1.shape(0)]) ; 
		    cuv::transpose(srcT, v1);


		    //TODO src transponieren
		    spn_prod_w_grad(*ptr2, r0.delta.cdata(), srcT, p2.value.cdata(), m_lae.cdata()); 
		    p2.push(ptr2);
	    }


                p1.value.reset();
                p2.value.reset();
                r0.delta.reset();
    }

}
