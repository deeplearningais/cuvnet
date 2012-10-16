#include "prod.hpp"

namespace cuvnet
{
    void Prod::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            // r0 = dot(p0,p1)
            cuv::prod(r0.overwrite_or_add_value().data(),
                    p0.value.cdata(),
                    p1.value.cdata(),
                    m_p0t, m_p1t);
        }else if(r0.can_add_directly()){
            // r0 += dot(p0,p1)
            cuv::prod(r0.overwrite_or_add_value().data(),
                    p0.value.cdata(),
                    p1.value.cdata(),
                    m_p0t, m_p1t,
                    1.f,1.f);
        }else{
            // allocate new value *sigh*
            value_ptr v(new value_type(r0.shape));
            cuv::prod(*v, 
                    p0.value.cdata(),
                    p1.value.cdata(),
                    m_p0t, m_p1t);
            r0.push(v);
        }
        if(!p0.need_derivative) p1.value.reset();
        if(!p1.need_derivative) p0.value.reset();
    }

    void Prod::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative || p1.need_derivative);
        if(p0.need_derivative){
            const value_type& delta = r0.delta.cdata();
            const value_type& p1v   = p1.value.cdata();
            char p1t    = (m_p0t==m_p1t) ? 't':'n';
            char deltat = (m_p0t=='t')   ? 't':'n';
            if(p0.can_overwrite_directly()){
                if(m_p0t=='n')
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            delta, p1v, deltat, p1t,1.f,0.f);
                else
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            p1v, delta, p1t, deltat,1.f,0.f);
            }
            else if(p0.can_add_directly()){
                if(m_p0t=='n')
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            delta, p1v, deltat, p1t,1.f,1.f);
                else
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            p1v, delta, p1t, deltat,1.f,1.f);
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(p0.shape));
                if(m_p0t=='n')
                    cuv::prod(v.data(), delta, p1v,
                            deltat, p1t,1.f,0.f);
                else
                    cuv::prod(v.data(), p1v, delta,
                            p1t, deltat,1.f,0.f);
                p0.push(v);
            }
        }
        if(p1.need_derivative){
            const value_type& delta = r0.delta.cdata();
            const value_type& p0v   = p0.value.cdata();
            char p0t    = (m_p0t==m_p1t) ? 't':'n';
            char deltat = (m_p1t=='t')   ? 't':'n';
            if(p1.can_overwrite_directly()){
                if(m_p1t=='n')
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            p0v, delta, p0t, deltat,1.f,0.f);
                else
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            delta,p0v, deltat, p0t,1.f,0.f);
            }
            else if(p1.can_add_directly()){
                if(m_p1t=='n')
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            p0v, delta, p0t,deltat,1.f,1.f);
                else
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            delta,p0v, deltat, p0t,1.f,1.f);
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(p1.shape));
                if(m_p1t=='n')
                    cuv::prod(v.data(),
                            p0v, delta, p0t,deltat,1.f,0.f);
                else
                    cuv::prod(v.data(),
                            delta,p0v, deltat,p0t,1.f,0.f);
                p1.push(v);
            }
        }
        r0.delta.reset();
    }

    void Prod::_determine_shapes(){
        param_t&  p0 = m_params[0];
        param_t&  p1 = m_params[1];

        unsigned int n = m_p0t=='n' ? p0->shape[0] : p0->shape[1];
        unsigned int m = m_p1t=='n' ? p1->shape[1] : p1->shape[0];

#ifndef NDEBUG
        unsigned int k0 = m_p0t=='n' ? p0->shape[1] : p0->shape[0];
        unsigned int k1 = m_p1t=='n' ? p1->shape[0] : p1->shape[1];
        assert(k0==k1);
#endif
        m_results[0]->shape.resize(2);
        m_results[0]->shape[0] = n;
        m_results[0]->shape[1] = m;
    }

}
