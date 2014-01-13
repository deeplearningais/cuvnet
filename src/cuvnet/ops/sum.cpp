#include "sum.hpp"

namespace cuvnet
{
    void Sum::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(m_identity) {
            r0.push(p0.value);
            p0.value.reset();
            return;
        }

        //#ifndef CUVNET_PRECISE_SUM
#if 1
        float sum = cuv::sum(p0.value.cdata());
#else
        float sum = kahan_summation(p0.value.cdata()); // this is expensive!!! use only for testing.
#endif
        if(r0.can_overwrite_directly()){
            (*r0.overwrite_or_add_value())[0] = sum;
        }
        else if(r0.can_add_directly()){
            (*r0.overwrite_or_add_value())[0] += sum;
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            v.data()[0] = sum;
            r0.push(v);
        }
        // don't delete p0, instead overwrite it in bprop
    }

    void Sum::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(m_identity){
            p0.push(r0.delta);
            r0.delta.reset();
            return;
        }

        if(p0.can_overwrite_directly()){
            value_ptr& v = p0.overwrite_or_add_value();
            v  = p0.value;     // only ptr is copied
            p0.value.reset();  // try overwriting p0
            v.data_onlyshape() = r0.delta.cdata()[0];
        }else if(p0.can_add_directly()){
            value_ptr& v = p0.overwrite_or_add_value();
            *v += (float)r0.delta.cdata()[0];
            p0.value.reset(); // try overwriting p0
        }else{
            value_ptr v = p0.value; // try overwriting p0
            p0.value.reset();
            *v = (float)r0.delta.cdata()[0];
            p0.push(v);
        }
        //r0.delta.reset(); // do not reset delta, it is very small anyway
    }

    void Sum::_determine_shapes(){
        m_results[0]->shape.resize(1);
        m_results[0]->shape[0] = 1;

        std::vector<unsigned int>& v = m_params[0]->shape;
        unsigned int s = std::accumulate(v.begin(),v.end(),1,std::multiplies<unsigned int>());
        if(s==1)
            m_identity = true;
    }





    void Mean::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(m_identity) {
            r0.push(p0.value);
            p0.value.reset();
            return;
        }

        float mean = cuv::mean(p0.value.cdata());
        if(r0.can_overwrite_directly()){
            (*r0.overwrite_or_add_value())[0] = mean;
        }
        else if(r0.can_add_directly()){
            (*r0.overwrite_or_add_value())[0] += mean;
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            v.data()[0] = mean;
            r0.push(v);
        }
        // don't delete p0, instead overwrite it in bprop
    }

    void Mean::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(m_identity){
            p0.push(r0.delta);
            r0.delta.reset();
            return;
        }

        if(p0.can_overwrite_directly()){
            value_ptr& v = p0.overwrite_or_add_value();
            v = p0.value;
            p0.value.reset(); // try overwriting p0
            *v =  m_div * r0.delta.cdata()[0];
        }else if(p0.can_add_directly()){
            value_ptr& v = p0.overwrite_or_add_value();
            *v += m_div * r0.delta.cdata()[0];
            p0.value.reset(); // try overwriting p0
        }else{
            value_ptr v = p0.value; // try overwriting p0
            p0.value.reset();
            *v = m_div * r0.delta.cdata()[0];
            p0.push(v);
        }
        //r0.delta.reset(); // do not reset delta, it is very small anyway
    }

    void Mean::_determine_shapes(){
        m_results[0]->shape.resize(1);
        m_results[0]->shape[0] = 1;
        std::vector<unsigned int>& v = m_params[0]->shape;
        unsigned int s = std::accumulate(v.begin(),v.end(),1,std::multiplies<unsigned int>());
        m_div = 1.f / s;

        if(s==1)
            m_identity = true;
    }
}
