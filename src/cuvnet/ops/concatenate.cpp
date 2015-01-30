
#include "cuvnet/tools/logging.hpp"
#include "concatenate.hpp"

namespace{
        log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("concatenate"));
}
namespace cuvnet
{
    template<class T>
    struct view_of{
        typedef cuv::tensor_view<typename T::value_type, typename T::memory_space_type, typename T::memory_layout_type> type;
    };

    void copy_center_dim_fprop(matrix& dst, const matrix& src){
        // copies a small NxM tensor into a
        // Nx1xM tensor (which is part of a NxKxM tensor)
        
        // this uses N copy operations of complexity O(M)
        for (unsigned int n = 0; n < dst.shape(0); ++n)
        {
            bool res = dst[cuv::indices[n]].copy_memory(src[cuv::indices[n]], false, 0);
            cuvAssert(res);
        }
    }
    void copy_center_dim_bprop(matrix& dst, const matrix& src){
        // copies a Nx1xM view into an NxM tensor
        // this uses N copy operations of complexity O(M)
        for (unsigned int n = 0; n < dst.shape(0); ++n)
        {
            dst[cuv::indices[n]] = src[cuv::indices[n]];
        }
    }

    std::vector<unsigned int> Concatenate::get_pi_shape(value_type & vi){
        std::vector<unsigned int> pi_shape;
        if (m_dim == 0) {
            pi_shape.resize(2);
            pi_shape[0] = vi.shape(m_dim); 
            pi_shape[1] = m_tmp_shape[1];
        }else if (m_dim < (m_params[0]->shape.size() -1)){
            pi_shape.resize(3);
            pi_shape[0] = m_tmp_shape[0]; 
            pi_shape[1] = vi.shape(m_dim);
            pi_shape[2] = m_tmp_shape[2];                        
        } else{
            pi_shape.resize(2);
            pi_shape[0] = m_tmp_shape[0]; 
            pi_shape[1] = vi.shape(m_dim);                        
        }
        return pi_shape;
    }

    Concatenate::value_type Concatenate::get_subtensor(const value_type &v, unsigned int position){
        unsigned int start = 0;
        for ( unsigned int i = 0; i < position; i++){
            start += m_pi_shape[i][m_dim];
        }  
        unsigned int end = start + m_pi_shape[position][m_dim];  

        if(m_dim == 0)  return v[cuv::indices[cuv::index_range(start, end)]];
        else            return v[cuv::indices[cuv::index_range()][cuv::index_range(start, end)]];

    }

    void Concatenate::fprop(){
        using namespace cuv;
        result_t::element_type& r0 = *m_results[0];

        if(r0.can_overwrite_directly()){
            value_type& v = *r0.overwrite_or_add_value(); // NOTE: copies meta-info!
            if (m_reshape) v.reshape(  m_tmp_shape );    // ... so we can reshape without reshaping back
            
            for (unsigned int i = 0; i < m_n; i++){
                param_t::element_type&  pi = *m_params[i];
                value_type dst_i = get_subtensor(v, i);
                //reshape input i
                value_type vi = pi.value.cdata();
                if (m_reshape) {
                    //get desired shape
                    std::vector<unsigned int> pi_shape = get_pi_shape(vi);                     
                    vi.reshape( pi_shape );
                }
                //LOG4CXX_WARN(g_log, "v" << i <<" has_nan: " << cuv::has_nan(vi));
                //LOG4CXX_WARN(g_log, "v" << i <<" has_inf: " << cuv::has_inf(vi));
                if(vi.ndim() < 3){
                    //LOG4CXX_WARN(g_log, "copy_otherdim");
                    bool res = dst_i.copy_memory(vi, false, 0);                
                    cuvAssert(res);
                } else{
                    //LOG4CXX_WARN(g_log, "copy_center_dim_fprop");
                    copy_center_dim_fprop(dst_i, vi);
                }
            }
            //LOG4CXX_WARN(g_log, "r" << " has_nan: " << cuv::has_nan(v));
            //LOG4CXX_WARN(g_log, "r" << " has_inf: " << cuv::has_inf(v));
            if(m_reshape) v.reshape(r0.shape);
        }else{
            value_ptr v1 = value_ptr(new value_type(r0.shape, value_ptr::s_allocator));
            
            value_type& v = *v1;
            if (m_reshape){ 
                v.reshape(  m_tmp_shape );
            }
            
            for (unsigned int i = 0; i < m_n; i++){        
                param_t::element_type&  pi = *m_params[i];                      
                value_type dst_i = get_subtensor(v, i);   
                //reshape input i
                value_type vi = pi.value.cdata();
                //LOG4CXX_WARN(g_log, "v" << i <<" nan:" << cuv::has_nan(vi) << " inf:" << cuv::has_inf(vi) << " max:"<<cuv::maximum(vi));
                if (m_reshape) {
                    //get desired shape
                    std::vector<unsigned int> pi_shape = get_pi_shape(vi);                     
                    vi.reshape( pi_shape );
                }
                if(vi.ndim() < 3){
                    //LOG4CXX_WARN(g_log, "copy_otherdim");
                    bool res = dst_i.copy_memory(vi, false, 0);                
                    cuvAssert(res);
                }else{
                    //LOG4CXX_WARN(g_log, "copy_center_dim_fprop");
                    copy_center_dim_fprop(dst_i, vi);
                }
            }
            //LOG4CXX_WARN(g_log, "r nan:" << cuv::has_nan(v) << " inf:" << cuv::has_inf(v) << " max:"<<cuv::maximum(v));
            v.reshape(  r0.shape );
            r0.push(v1);
        }

        // reset all params
        for ( unsigned int i = 0; i < m_n; i++){
            param_t::element_type&  pi = *m_params[i];
            pi.value.reset();
        }
    }

    
    void Concatenate::bprop(){
        using namespace cuv;
        result_t::element_type& r0 = *m_results[0];
        
        //assertion, we should need a derivative
        bool need_derivative = false;
        for ( unsigned int i = 0; i < m_n; i++){
            param_t::element_type&  pi = *m_params[i];
            need_derivative = (pi.need_derivative || need_derivative);
            if ( need_derivative ) break;
        }
        assert(need_derivative);
  
        value_type v = r0.delta.cdata();
        if (m_reshape) v.reshape(  m_tmp_shape );
        
        for ( unsigned int i = 0; i < m_n; i++){
            param_t::element_type&  pi = *m_params[i];
            if(pi.need_derivative){
                if(pi.can_overwrite_directly()){            
                    value_type vi = *pi.overwrite_or_add_value();
                    if (m_reshape) {
                        //get desired shape
                        std::vector<unsigned int> pi_shape = get_pi_shape(vi);                     
                        assert(vi.is_c_contiguous());
                        vi.reshape( pi_shape );
                    }
                    if(vi.ndim() < 3){
                        vi.copy_memory(get_subtensor(v, i), false, 0);
                    }else{
                        copy_center_dim_bprop(vi, get_subtensor(v,i));
                    }
                }else if(pi.can_add_directly() && m_pi_shape[0].size() < 3){
                    value_type dsti = *pi.overwrite_or_add_value();
                    if (m_reshape) {
                        //get desired shape
                        std::vector<unsigned int> pi_shape = get_pi_shape(dsti);                     
                        assert(dsti.is_c_contiguous());
                        dsti.reshape( pi_shape );
                    }
                    value_type vi = get_subtensor(v, i);
                    if(!vi.is_c_contiguous())
                        dsti += vi.copy();
                    else
                        dsti += vi;
                }else{
                    value_ptr vd;
                    matrix tmp = get_subtensor(v,i);
                    if(tmp.ndim()<3){
                        vd.reset(new value_type(tmp.copy()));
                    }else{
                        vd.reset(new value_type(tmp.shape()));
                        copy_center_dim_bprop(*vd, tmp);
                    }
                    if(m_reshape){
                        assert(vd->is_c_contiguous());
                        vd->reshape(pi.shape);
                    }
                    pi.push(vd);
                }            
            }
        }
        r0.delta.reset();
    }

    void Concatenate::_determine_shapes(){
        param_t&  p0 = m_params[0];
        unsigned int size = p0->shape.size();
        
        //if ( ( m_dim != 0 ) && ( m_dim != size -1) ) 
            //throw std::runtime_error("This type of concatenation is not yet implemented ( since the copy memory operation is not yet implemented)\nIf memcopy for generic shapes is implemented now, please just remove this assertion\n");
        
        //assert that all concat elements have the same size
        for ( unsigned int i = 1; i < m_params.size(); i++){
            cuvAssert( m_params[0]->shape.size() == m_params[i]->shape.size() );
            //std::cout << "Concatenate: Comparing parameter:" << i << std::endl;
            //arrays must have same shape (except in dimension m_dim, along which we concatenate)
            for (unsigned int j = 0; j < m_params[0]->shape.size(); j++){
                //std::cout << "... j:" << j << " m_params[0]->shape[j]:" << m_params[0]->shape[j] << " m_params[i]->shape[j]:" << m_params[i]->shape[j] << std::endl;
                if ( j != m_dim)
                    cuvAssert(m_params[0]->shape[j] == m_params[i]->shape[j] );
            }
        }
                   
        m_pi_shape.resize( m_n, std::vector<int>( size , 0) );
        
        for (unsigned int i = 0; i < m_n; ++i){
            param_t&  pi = m_params[i];
            for ( unsigned int s = 0; s < size; s++){
                m_pi_shape[i][s] = pi->shape[s];
            }
        }
        
        m_results[0]->shape.resize(size);
        for(unsigned int i = 0; i < size; i++){
            if(i == m_dim){
                unsigned int n = 0;
                for ( unsigned int s = 0; s < m_n; s++)
                    n += m_pi_shape[s][m_dim];
                m_results[0]->shape[i] = n;
            }else{
                m_results[0]->shape[i] = p0->shape[i];
            }
        }
        
        //calculate shape of temp tensor if dim > 3
        if ( size > 2){
             m_reshape = true;
            // multiply all axes before m_dim
             unsigned int before = 1;
             unsigned int after  = 1;
             //assumption: shapes before m_dim must be the same
             for ( unsigned int i = 0; i < m_dim; i++) 
                 before *= p0->shape[i];
             //assumption: shapes after m_dim must be the same
             for ( unsigned int i = m_dim+1; i < size; i++) 
                 after *= p0->shape[i];
             

             //compute new m_dim
             if (m_dim == 0) {
                m_tmp_shape.resize(2);
                m_tmp_shape[0] = m_results[0]->shape[0];;
                m_tmp_shape[1] = after;
                 
            } else if ( m_dim == size -1) {
                m_tmp_shape.resize(2);
                m_tmp_shape[0] = before;
                m_tmp_shape[1] = m_results[0]->shape[m_dim];
            } else{
                m_tmp_shape.resize(3);
                m_tmp_shape[0] = before;
                m_tmp_shape[1] = m_results[0]->shape[m_dim];
                m_tmp_shape[2] = after;
            }
        } else
            m_reshape = false;
        
    }
}
