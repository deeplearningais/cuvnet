#include <cuv/convolution_ops/convolution_ops.hpp>
#include "reshape.hpp"

namespace cuvnet
{
    void Flatten::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            value_type& v = *r0.overwrite_or_add_value();
            if(m_copy){
                v = p0.value.data(); // this is always fine, but may take more time than no-copy.
            }else{
                v = p0.value.cdata(); // this is O(1), but violates const-correctness(!)
            }
            v.reshape(r0.shape);
        }else{
            value_ptr v;
            if(m_copy)
                v.reset(new value_type(p0.value.data())); //this is always fine, but may take more time than no-copy
            else
                v.reset(new value_type(p0.value.cdata())); // this is O(1), but violates const-correctness(!)
            v->reshape(r0.shape);
            r0.push(v);
        }
        p0.value.reset();
    }
    void Flatten::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            value_type& v = *p0.overwrite_or_add_value();
            if(m_copy)
                v = r0.delta.data(); // O(n) if someone else wants to read still, but always safe
            else
                v = r0.delta.cdata(); // O(1), but violates const-correctness again!
            v.reshape(p0.shape);
        }else{
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            if(m_copy)
                *v = r0.delta.data(); // O(n) if someone else wants to read still, but always safe
            else
                *v = r0.delta.cdata(); // O(1), but violates const-correctness
            v->reshape(p0.shape);
            p0.push(v);
        }
        r0.delta.reset();
    }
    void Flatten::_determine_shapes(){
        assert(m_params[0]->shape.size() >= m_outdim);
        std::vector<unsigned int> p0 = m_params[0]->shape;
        std::vector<unsigned int> dst(m_outdim);
        for(unsigned int i=0;i<m_outdim-1;i++)
            dst[i] = p0[i];
        unsigned int size = 1;
        for(unsigned int i=m_outdim-1;i<p0.size();i++)
            size *= p0[i];
        dst[m_outdim-1] = size;
        m_results[0]->shape = dst;
    }

    void Reshape::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            value_type& v = *r0.overwrite_or_add_value();
            if(m_copy)
                v = p0.value.data(); // this safer but may be slower
            else
                v = p0.value.cdata(); // this is O(1), but violates const-correctness(!)
            v.reshape(r0.shape);
        }else{
            value_ptr v;
            if(m_copy)
                v.reset(new value_type(p0.value.data())); // this safer but maybe slower
            else
                v.reset(new value_type(p0.value.cdata())); // this is O(1), but violates const-correctness(!)
            v->reshape(r0.shape);
            r0.push(v);
        }
        p0.value.reset();
    }

    void Reshape::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            value_type& v = *p0.overwrite_or_add_value();
            if(m_copy)
                v = r0.delta.cdata().copy(); // safe but slow
            else
                v = r0.delta.cdata(); // O(1), but violates const-correctness again!
            v.reshape(p0.shape);
        }else{
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            if(m_copy)
                *v = r0.delta.cdata().copy(); // safe but slow
            else
                *v = r0.delta.cdata(); // O(1), but violates const-correctness
            v->reshape(p0.shape);
            p0.push(v);
        }
        r0.delta.reset();
    }

    void Reshape::_determine_shapes(){
        std::vector<unsigned int> p0 = m_params[0]->shape;

        int special = 0;
        for (unsigned int i = 0; i < m_shape.size(); ++i){
            cuvAssert(m_shape[i]!=0);
            special += m_shape[i]<0;
        }
        if(!special){
            // no negative values
            m_results[0]->shape.clear();
            m_results[0]->shape.reserve(m_shape.size());
            std::copy(m_shape.begin(), m_shape.end(), std::back_inserter(m_results[0]->shape));
            return;
        }
        cuvAssert(special==1); // only works if /one/ dimension must be deduced
        std::vector<unsigned int> dst(m_shape.size());
        unsigned int n_in  =  std::accumulate(p0.begin(),p0.end(),1,std::multiplies<unsigned int>());
        unsigned int n_out = -std::accumulate(m_shape.begin(),m_shape.end(),1,std::multiplies<int>());
        cuvAssert(n_in%n_out == 0);
        for (unsigned int i = 0; i < m_shape.size(); ++i){
            if(m_shape[i]>0) dst[i] = m_shape[i];
            else             dst[i] = n_in/n_out;
        }

        m_results[0]->shape = dst;
    }
}
