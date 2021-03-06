#include "sum_out_dim.hpp"
#include <cuvnet/tools/matwrite.hpp>

namespace cuvnet
{
    void SumOutDim::_graphviz_node_desc(detail::graphviz_node& desc)const{

        if(m_axis == 0)
            desc.label = "sum_out..col";
        else if(m_axis == 1)
            desc.label = "sum_out..row";
        else 
            desc.label = "sum_out.." + boost::lexical_cast<std::string>(m_axis);
        if(m_mean){
            desc.label += " (mean)";
        }
        if(m_squared){
            desc.label += " (squared)";
        }
        if(m_lae){
            desc.label += " (lae)";
        }
    }

    void SumOutDim::fprop(){
        using namespace cuv;

        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        float fact_new = m_mean ? 1.f/m_n_summed : 1.f;
        reduce_functor red_func = m_squared ? RF_ADD_SQUARED : RF_ADD;
	if ( m_lae ){
		red_func = RF_LOGADDEXP;
	}
        //reshape param to 2 dims
        value_type v0 = p0.value.cdata();
        v0.reshape( m_param_reshape );
        // sum up all squared entries        
        if(r0.can_overwrite_directly()){
            // reshape result tensor for operations
            value_type v1 = *r0.overwrite_or_add_value();
            v1.reshape(  m_res_reshape );
            
            if(m_axis == 0) cuv::reduce_to_row(v1, v0, red_func, fact_new, 0.f);
            else            cuv::reduce_to_col(v1, v0, red_func, fact_new, 0.f);
            if ( m_lae )
                m_lae_res = v1;	    
        } else if(r0.can_add_directly()){
            // reshape result tensor for operations
            value_type v1 = *r0.overwrite_or_add_value();
            v1.reshape(  m_res_reshape );
            
            if(m_axis == 0) cuv::reduce_to_row(v1, v0,red_func,fact_new,1.f);
            else            cuv::reduce_to_col(v1, v0,red_func,fact_new,1.f);
	    if ( m_lae )
		    m_lae_res = v1;
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type( m_res_reshape, value_ptr::s_allocator));
            if(m_axis == 0) cuv::reduce_to_row(*v, v0, red_func, fact_new, 0.f);
            else            cuv::reduce_to_col(*v, v0, red_func, fact_new, 0.f);
            v->reshape (  m_res_shape );
            r0.push(v);
            if ( m_lae )
		    m_lae_res = v;
        }

        if(m_squared || m_lae){
            if(!p0.need_derivative)
                p0.value.reset(); // needed for bprop
        }else{
            p0.value.reset(); 
        }
    }

    // todo lae grad 
    void SumOutDim::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        float fact_new = m_mean ? 1.f/m_n_summed : 1.f;
        assert(p0.need_derivative);
        
        //reshape delta
        value_type r = r0.delta.cdata();
        r.reshape( m_res_reshape );
        
        unsigned int axis;
        if (m_axis == 0) 
		axis = 1;
        else axis = 0;
        if(!m_squared && !m_lae){
            if(p0.can_overwrite_directly()){
                //reshape params to 2d tensor
                value_type p0_data = *p0.overwrite_or_add_value();
                p0_data.resize( m_param_reshape);
		    matrix_op_vec(
                        p0_data, 
                        p0_data, 
                         r, axis, BF_2ND, fact_new, 0.f);             
            }else if(p0.can_add_directly()){
                //reshape params to 2d tensor
                value_type p0_data = *p0.overwrite_or_add_value();
                p0_data.resize( m_param_reshape);
		    matrix_op_vec(
                        p0_data, 
                        p0_data, 
                        r, axis, BF_ADD, fact_new, 1.f);           
            }else{
                value_ptr v(new value_type( m_param_reshape, value_ptr::s_allocator));
	  	    matrix_op_vec(
                        *v, *v, 
                        r, axis, BF_2ND, fact_new, 0.f);
                v->reshape( m_param_shape);
                p0.push(v);
            }
        }else{
            if ( m_lae ){
                //reshape params to 2d tensor
                value_type p0_data = p0.value.cdata();
                p0_data.resize( m_param_reshape);
            
                value_type p0_dst = p0.value.data_onlyshape();
                p0_dst.resize( m_param_reshape);            
         

                //reshape result
                value_type res = m_lae_res.cdata();
                res.reshape( m_res_reshape );
		
                // try to overwrite p0
                const value_type& p0value = p0_data;
                value_type& dst = p0_dst;
		
		//calculate grad of lae ( exp(x)/ sum (exp(x)))
                apply_scalar_functor(res, res, SF_EXP);
		apply_scalar_functor(dst, p0value, SF_EXP);
               
	       	matrix_op_vec(
                        dst,
                        dst,
                        res, axis, BF_DIV, fact_new, 0.f);
                //multiply with delta
	       	matrix_op_vec(
                        dst,
                        dst,
                        r, axis, BF_MULT, fact_new, 0.f);
               
	       	p0.push(p0.value);
                //tofile("lae_res.dat", p0.value.cdata());
                p0.value.reset();

            } else {

                //reshape params to 2d tensor
                value_type p0_data = p0.value.cdata();
                p0_data.resize( m_param_reshape);
            
                value_type p0_dst = p0.value.data_onlyshape();
                p0_dst.resize( m_param_reshape);            
            
                // try to overwrite p0
                const value_type& p0value = p0_data;
                value_type& dst = p0_dst;

                apply_scalar_functor(dst, p0value, SF_MULT, 2.f); // ideally in-place
                matrix_op_vec(
                        dst,
                        dst,
                        r, axis, BF_MULT, fact_new, 0.f);
                p0.push(p0.value);
                p0.value.reset();
	    }
        }
    }

    
    void SumOutDim::_determine_shapes(){
        param_t::element_type&  p0 = *m_params[0];
        
        m_ndim = p0.shape.size();
        assert(m_ndim >= 2);
        //assert first or last axis
        assert((m_axis == m_ndim-1) || m_axis == 0);

        m_res_reshape.resize(1);
        
        if (m_mean)m_n_summed = (float)p0.shape[m_axis];
            
        m_param_reshape.resize(2);
        m_param_reshape[0] = p0.shape[0];
        m_param_reshape[1] = p0.shape[m_ndim - 1];

        m_res_shape.resize(m_ndim);
        m_param_shape.resize(m_ndim);
        //remember shape, for reshape

        if (m_ndim > 2){
            for ( unsigned int i = 0; i < m_ndim; i++) {
                m_param_shape[i] = p0.shape[i];
                m_res_shape[i]   = p0.shape[i]; 
            }
            m_res_shape[m_axis] = 1;
            //get shape for sum operation
            if (m_axis == 0){
                m_res_reshape[0] =  p0.shape[m_ndim - 1];
                for ( unsigned int i = 1; i < m_ndim -1; i++) {
                    m_param_reshape[1] *= p0.shape[i];
                    m_res_reshape[0]   *= p0.shape[i];
                }
            } else {
                m_res_reshape[0] = p0.shape[0];
                for ( unsigned int i = 1; i < m_ndim -1; i++){ 
                    m_param_reshape[0] *= p0.shape[i];                
                    m_res_reshape[0]   *= p0.shape[i];   
                }
            }            
	} else {
		for ( unsigned int i = 0; i < m_ndim; i++) {
			m_param_shape[i] = p0.shape[i];
			m_res_shape[i]   = p0.shape[i]; 
		}
		if (m_axis == 0){
			m_res_reshape[0] = p0.shape[m_ndim -1];       
			m_res_shape[m_axis] = 1;
		} else{
			m_res_reshape[0] = p0.shape[0];       
			m_res_shape[m_axis] = 1;
		}
	}

        m_results[0]->shape.resize(m_ndim);
        for ( unsigned int i = 0; i < m_ndim; i++){
            if ( i == m_axis ) m_results[0]->shape[i] = 1;
            else               m_results[0]->shape[i] = p0.shape[i];
        }

    }


}
