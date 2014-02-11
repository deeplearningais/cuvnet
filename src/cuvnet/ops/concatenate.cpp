
#include "concatenate.hpp"

namespace cuvnet
{

    template<class T>
    struct view_of{
        typedef cuv::tensor_view<typename T::value_type, typename T::memory_space_type, typename T::memory_layout_type> type;
    };

    Concatenate::value_type Concatenate::get_subtensor(const value_type &v, bool first){
        if(m_dim == 0){
            if(first){
                return v[cuv::indices[cuv::index_range(0, m_p0_shape[0])]];
            }else{
                return v[cuv::indices[cuv::index_range(m_p0_shape[0], m_p0_shape[0] + m_p1_shape[0])]];
            }
        }
        else if(m_dim == 1){
            if(first){
                return v[cuv::indices[cuv::index_range()][cuv::index_range(0, m_p0_shape[1])]];
            }else{
                return v[cuv::indices[cuv::index_range()][cuv::index_range(m_p0_shape[1], m_p0_shape[1] + m_p1_shape[1])]];
            }
        }
        else{ 
            if(first){
                return v[cuv::indices[cuv::index_range()][cuv::index_range()][cuv::index_range(0, m_p0_shape[2])]];
            }else{
                return v[cuv::indices[cuv::index_range()][cuv::index_range()][cuv::index_range(m_p0_shape[2], m_p0_shape[2] + m_p1_shape[2])]];
            }
        }
    }

    void Concatenate::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            value_type& v = *r0.overwrite_or_add_value();
            value_type part_1 = get_subtensor(v, true);
            *static_cast<view_of<value_type>::type*>(&part_1) = p0.value.cdata();

            value_type part_2 = get_subtensor(v, false);
            *static_cast<view_of<value_type>::type*>(&part_2)  = p1.value.cdata();
        }else{
            value_ptr v = value_ptr(new value_type(r0.shape, value_ptr::s_allocator)); // this safer but slower
            value_type part_1 = get_subtensor(*v, true);
            *static_cast<view_of<value_type>::type*>(&part_1) = p0.value.cdata();

            value_type part_2 = get_subtensor(*v, false);
            *static_cast<view_of<value_type>::type*>(&part_2) = p1.value.cdata();
            r0.push(v);
        }
        if(!p0.need_derivative) p1.value.reset();
        if(!p1.need_derivative) p0.value.reset();
    }

    void Concatenate::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative || p1.need_derivative);

        if(p0.need_derivative){
            if(p0.can_overwrite_directly()){
                value_type& v0 = *p0.overwrite_or_add_value();
                const value_type& v = r0.delta.cdata();
                *static_cast<view_of<value_type>::type*>(&v0) = get_subtensor(v, true);
            }else if(p0.can_add_directly()){
                value_type& v0 = *p0.overwrite_or_add_value();
                const value_type& v = r0.delta.cdata();
                v0 += get_subtensor(v, true);
            }else{
                value_ptr v0(new value_type(p0.shape, value_ptr::s_allocator));

                const value_type& v = r0.delta.cdata();
                *static_cast<view_of<value_type>::type*>(&*v0) = get_subtensor(v, true);
                p0.push(v0);
            }

        }
        if(p1.need_derivative){
            if(p1.can_overwrite_directly()){
                value_type& v1 = *p1.overwrite_or_add_value();
                const value_type& v = r0.delta.cdata();
                *static_cast<view_of<value_type>::type*>(&v1) = get_subtensor(v, false);
            }else if(p1.can_add_directly()){
                value_type& v1 = *p1.overwrite_or_add_value();
                const value_type& v = r0.delta.cdata();
                v1 += get_subtensor(v, false);
            }else{
                value_ptr v1(new value_type(p1.shape, value_ptr::s_allocator));

                value_type v = r0.delta.cdata();
                *static_cast<view_of<value_type>::type*>(&*v1) = get_subtensor(v, false);
                p1.push(v1);
            }

        }
        r0.delta.reset();
    }

    void Concatenate::_determine_shapes(){
        param_t&  p0 = m_params[0];
        param_t&  p1 = m_params[1];
        unsigned int size = p0->shape.size();
        m_p0_shape = std::vector<int>(size);
        m_p1_shape = std::vector<int>(size);
        for (unsigned int i = 0; i < size; ++i)
        {
            m_p0_shape[i] = p0->shape[i];
            m_p1_shape[i] = p1->shape[i];
        }
        
        m_results[0]->shape.resize(size);
        for(unsigned int i = 0; i < size; i++){
            if(i == m_dim){
                m_results[0]->shape[i] = p0->shape[i] + p1->shape[i];
            }else{
                m_results[0]->shape[i] = p0->shape[i];
            }
        }
    }
/////////////////////////////////////////////////////////////////////////////// concatenate_n ////////////////////////////////////////////////////////////////

    Concatenate_N::value_type Concatenate_N::get_subtensor(const value_type &v, unsigned int position){
        unsigned int start = 0;
        for ( unsigned int i = 0; i < position; i ++) 
            start += m_pi_shape[i][m_dim];
        unsigned int end = start + m_pi_shape[position][m_dim];
        
        if(m_dim == 0)      return v[cuv::indices[cuv::index_range(start, end)]];
        else if(m_dim == 1) return v[cuv::indices[cuv::index_range()][cuv::index_range(start, end)]];
        else                return v[cuv::indices[cuv::index_range()][cuv::index_range()][cuv::index_range(start, end)]];
    }

    void Concatenate_N::fprop(){
        using namespace cuv;
        result_t::element_type& r0 = *m_results[0];

        if(r0.can_overwrite_directly()){
            value_type& v = *r0.overwrite_or_add_value();       
            for (unsigned int i = 0; i < m_n; i++){
                param_t::element_type&  pi = *m_params[i];
                value_type part_i = get_subtensor(v, i);
                *static_cast<view_of<value_type>::type*>(&part_i) = pi.value.cdata();                
            }
        }else{
            value_ptr v = value_ptr(new value_type(r0.shape, value_ptr::s_allocator)); // this safer but slower
                  
            for (unsigned int i = 0; i < m_n; i++){        
                param_t::element_type&  pi = *m_params[i];                      
                value_type part_i = get_subtensor(v, i);                     
                *static_cast<view_of<value_type>::type*>(&part_i) = pi.value.cdata();                
            }
            r0.push(v);
        }

        // reset all params
        for ( unsigned int i = 0; i < m_n; i++){
            param_t::element_type&  pi = *m_params[i];
            if (! pi.need_derivative) pi.value.reset();
        }
    }

    void Concatenate_N::bprop(){
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
        
        for ( unsigned int i = 0; i < m_n; i++){
            param_t::element_type&  pi = *m_params[i];
            if(pi.need_derivative){
                if(pi.can_overwrite_directly()){
                    value_type& vi = *pi.overwrite_or_add_value();
                    const value_type& v = r0.delta.cdata();
                    *static_cast<view_of<value_type>::type*>(&vi) = get_subtensor(v, i);
                }else if(pi.can_add_directly()){
                    value_type& vi = *pi.overwrite_or_add_value();
                    const value_type& v = r0.delta.cdata();
                    vi += get_subtensor(v, i);
                }else{
                    value_ptr vi(new value_type(pi.shape, value_ptr::s_allocator));

                    const value_type& v = r0.delta.cdata();
                    *static_cast<view_of<value_type>::type*>(&*vi) = get_subtensor(v, i);
                    pi.push(vi);
                }            
            }
        }
        r0.delta.reset();
    }

    
    void Concatenate_N::_determine_shapes(){
        param_t&  p0 = m_params[0];
        unsigned int size = p0->shape.size();
        m_pi_shape.resize( m_n, std::vector<int>( size , 0) );
        
        for (unsigned int i = 0; i < m_params.size(); ++i)
        {
            param_t&  pi = m_params[i];
            for ( unsigned int s = 0; s < size; s++){
                m_pi_shape[i][s] = pi->shape[s];
            }
            std::cout << std::endl;
        }
        
        m_results[0]->shape.resize(size);
        for(unsigned int i = 0; i < size; i++){
            if(i == m_dim){
                m_results[0]->shape[i]  = 0;
                for ( unsigned int s = 0; s < m_n; s++){
                    m_results[0]->shape[i] += m_pi_shape[s][i];
                }
            }else{
                m_results[0]->shape[i] = p0->shape[i];
            }
        }
    }
}
