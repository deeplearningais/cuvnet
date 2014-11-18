#include "noiser.hpp"
#include <boost/format.hpp>

namespace cuvnet
{
    void Noiser::release_data(){
        m_zero_mask.dealloc();
        Op::release_data();
    }
    
    void Noiser::_graphviz_node_desc(detail::graphviz_node& desc)const{
        std::string act = m_active ? "on" : "off";
        if(m_noisetype == NT_ZERO_OUT)
            desc.label = (boost::format("zero out noise %1.2f [%s]") % act % m_param).str();
        else if(m_noisetype == NT_NORMAL)
            desc.label = (boost::format("gaussian noise %1.2f [%s]") % act % m_param).str();
        else if(m_noisetype == NT_SALT_AND_PEPPER)
            desc.label = (boost::format("salt/pepper noise %1.2f [%s]") % act % m_param).str();
    }


    void Noiser::fprop_salt_and_pepper(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        // construct 2nd matrix with uniform values, binarize
        value_type rnd(p0.shape, get_global_allocator());

        // salt
        cuv::fill_rnd_uniform(rnd);
        m_zero_mask.resize(rnd.shape());
        cuv::apply_scalar_functor(m_zero_mask, rnd, SF_LT, m_param / 2.f);

        value_type&       res    = p0.value.data();
        cuv::apply_scalar_functor(res,SF_MULT,0.f,&m_zero_mask);

        // pepper (assumes maximum is 1)
        cuv::apply_scalar_functor(m_zero_mask, rnd, SF_GT, 1.f - m_param / 2.f);
        cuv::apply_scalar_functor(res, SF_RSUB, 1.f); // res = 1 - res
        cuv::apply_scalar_functor(res,SF_MULT,0.f,&m_zero_mask);
        cuv::apply_scalar_functor(res, SF_RSUB, 1.f); // res = 1 - res

        // do not compensate for salt/pepper

        r0.push(p0.value);
        p0.value.reset();
    }

    void Noiser::fprop_zero_out(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        //const value_type& inp0 = p0.value.cdata();           // original

        // construct 2nd matrix with uniform values, binarize
        value_type rnd(p0.shape, get_global_allocator());
        cuv::fill_rnd_uniform(rnd);
        m_zero_mask.resize(rnd.shape());
        cuv::apply_scalar_functor(m_zero_mask, rnd, SF_LT, m_param);

        value_type* res;
        if(m_mem_optimized){
            res    = p0.value.ptr();
            cuv::apply_scalar_functor(*res,SF_MULT,0.f,&m_zero_mask);
            m_zero_mask.dealloc();
        }else{
            res    = &p0.value.data();
            cuv::apply_scalar_functor(*res,SF_MULT,0.f,&m_zero_mask);
        }

        if(m_compensate){
            // remaining units are "amplified", so that during
            // _inactive_ forward pass, the "mass" arriving at the next
            // layer is approximately the same.
            *res *= 1.f/(1.f - m_param); 
        }

        r0.push(p0.value);
        if(!m_mem_optimized  || !p0.need_derivative){
            p0.value.reset();
        }
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
            case NT_SALT_AND_PEPPER: fprop_salt_and_pepper(); break;
        }
    }

    void Noiser::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);
        assert(m_noisetype != NT_SALT_AND_PEPPER);


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

            if(m_mem_optimized){
                // determine values that were zeroed out by looking at our /input/
                m_zero_mask.resize(p0.shape);
                apply_scalar_functor(m_zero_mask, p0.value.cdata(), SF_EQ, 0.f);
            }

            // TODO does not account for compensation here!
            if(p0.can_add_directly()){
                // TODO: add masks for binary ops to CUV
                value_type& d_res = r0.delta.data_onlyshape();
                cuv::apply_scalar_functor(d_res,d_orig,SF_MULT,0.f,&m_zero_mask);
                p0.overwrite_or_add_value().data() += d_res;
            }else if(p0.can_overwrite_directly()){
                value_type& dst = *p0.overwrite_or_add_value();
                cuv::apply_scalar_functor(dst,d_orig,SF_COPY);
                cuv::apply_scalar_functor(dst,SF_MULT,0.f,&m_zero_mask);
            }else{
                // try to overwrite r0.delta
                value_type& d_res = r0.delta.data(); // COPY if not unique!
                cuv::apply_scalar_functor(d_res,d_orig,SF_MULT,0.f,&m_zero_mask);
                p0.push(r0.delta);
            }
        }
        p0.value.reset();
        r0.delta.reset();
    }
    void Noiser::_determine_shapes(){
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        r0.shape = p0.shape;

        m_mem_optimized = false;
        if(m_noisetype == NT_ZERO_OUT){
            if(p0.n_uses() == 1){
                if(p0.use(0)->m_single_result != p0.use(0)->result_uses.end())
                    m_mem_optimized = true;
            }
            if(m_force_mem_optimized)
                m_mem_optimized = true;
        }
    }
}
