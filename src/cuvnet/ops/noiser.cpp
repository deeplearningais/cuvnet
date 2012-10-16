#include "noiser.hpp"

namespace cuvnet
{
    void Noiser::release_data(){
        m_zero_mask.dealloc();
        Op::release_data();
    }


    void Noiser::fprop_zero_out(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        //const value_type& inp0 = p0.value.cdata();           // original

        // construct 2nd matrix with uniform values, binarize
        value_type rnd(p0.shape);
        cuv::fill_rnd_uniform(rnd);
        m_zero_mask.resize(rnd.shape());
        cuv::apply_scalar_functor(m_zero_mask, rnd, SF_LT, m_param);

        value_type&       res    = p0.value.data();
        cuv::apply_scalar_functor(res,SF_MULT,0.f,&m_zero_mask);

        // remaining units are "amplified", so that during
        // _inactive_ forward pass, the "mass" arriving at the next
        // layer is approximately the same.
        res *= 1.f/(1.f - m_param); 

        r0.push(p0.value);
        p0.value.reset();
    }

    void Noiser::fprop_normal(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        //const value_type& inp0 = p0.value.cdata();           // original

        if(r0.can_overwrite_directly()){
            value_ptr& v  = r0.overwrite_or_add_value();
            *v            = p0.value.cdata().copy(); 
            cuv::add_rnd_normal(*v,m_param);
        }
        else if(r0.can_add_directly()){
            value_ptr& v = r0.overwrite_or_add_value();
            *v += p0.value.cdata();
            cuv::add_rnd_normal(*v,m_param);
            p0.value.reset(); // forget it
        }
        else{
            value_ptr v = p0.value; // copy p0
            p0.value.reset();       // try to overwrite r0
            cuv::add_rnd_normal(*v,m_param);
            r0.push(v);
        }
        p0.value.reset();
    }

    void Noiser::fprop(){
        if(!m_active){
            param_t::element_type&  p0 = *m_params[0];
            result_t::element_type& r0 = *m_results[0];
            r0.push(p0.value);
            p0.value.reset();
            return;
        }
        switch(m_noisetype){
            case NT_NORMAL: fprop_normal(); break;
            case NT_ZERO_OUT: fprop_zero_out(); break;
        }
    }

    void Noiser::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);


        if(!m_active || m_noisetype == NT_NORMAL){
            if(p0.can_add_directly()){
                p0.overwrite_or_add_value().data() += r0.delta.cdata();
            }else if(p0.can_overwrite_directly()){
                p0.overwrite_or_add_value() = r0.delta;
            }else{
                p0.push(r0.delta);
            }
        }
        else if(m_noisetype == NT_ZERO_OUT){
            const value_type& d_orig = r0.delta.cdata();

            if(p0.can_add_directly()){
                // TODO: add masks for binary ops to CUV
                value_type& d_res = r0.delta.data_onlyshape();
                cuv::apply_scalar_functor(d_res,d_orig,SF_MULT,0.f,&m_zero_mask);
                p0.overwrite_or_add_value().data() += d_res;
            }else if(p0.can_overwrite_directly()){
                cuv::apply_scalar_functor(*p0.overwrite_or_add_value(),d_orig,SF_MULT,0.f,&m_zero_mask);
            }else{
                value_type& d_res = r0.delta.data_onlyshape();
                cuv::apply_scalar_functor(d_res,d_orig,SF_MULT,0.f,&m_zero_mask);
                p0.push(r0.delta);
            }
        }
        r0.delta.reset();
    }
}
