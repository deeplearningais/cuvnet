#include "upscale.hpp"
namespace cuvnet
{       
    void Upscale::fprop(){
        using namespace cuv;
        using namespace cuv::misc_conv;
        result_t::element_type& r0 = *m_results[0];
        param_t::element_type& p0 = *m_params[0]; 
        if (r0.can_overwrite_directly())
        {
            upscaleOp(*r0.overwrite_or_add_value(),p0.value.cdata(), factor);
        }
        else
        {
            // reallocate
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            upscaleOp(*v, p0.value.cdata(), factor);
            r0.push(v);

        }
        
        

        

        p0.value.reset();
    
    }
    void Upscale::bprop(){
        using namespace cuv;
        using namespace cuv::misc_conv;
        
        param_t::element_type& p0 = *m_params[0]; 
        result_t::element_type& r0 = *m_results[0]; 
        
        assert(p0.need_derivative);
        
        

        if (p0.can_overwrite_directly())
        {
            upscaleGrad(*p0.overwrite_or_add_value(),r0.delta.cdata(), factor);
        }
        else
        {
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            upscaleGrad(*v, r0.delta.cdata(), factor);
            p0.push(v);
        
        

        }
        r0.delta.reset();
    }
    void Upscale::_determine_shapes()
    {
        // determine shape of the only result
        result_t::element_type& r0 = *m_results[0]; 
        param_t::element_type& p0 = *m_params[0]; 
        std::vector<unsigned int> dst_shape(4);
        dst_shape[0] = p0.shape[0];
        dst_shape[1] = p0.shape[1]*factor;
        dst_shape[2] = p0.shape[2]*factor;
        dst_shape[3] = p0.shape[3];
        r0.shape = dst_shape;
    }
}

