#include <cuvnet/tools/logging.hpp>
#include "prod.hpp"

namespace cuvnet
{
    void Prod::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.color="chartreuse3";
        if (m_p0t == 't' && m_p1t == 't'){
           desc.label = "A' B'";
        }
        else if (m_p0t == 'n' && m_p1t == 't'){
           desc.label = "A B'";
        }
        else if (m_p0t == 't' && m_p1t == 'n'){
           desc.label = "A' B";
        }
        else if (m_p0t == 'n' && m_p1t == 'n'){
           desc.label = "A B";
        }
    }
    void Prod::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        value_type vp0 = p0.value.cdata();
        value_type vp1 = p1.value.cdata();
        vp0.reshape(m_sp0);
        vp1.reshape(m_sp1);
        if(r0.can_overwrite_directly()){
            r0.overwrite_or_add_value().data().reshape(m_sr);
            // r0 = dot(p0,p1)
            cuv::prod(r0.overwrite_or_add_value().data(),
                    vp0,
                    vp1,
                    m_p0t, m_p1t);
            r0.overwrite_or_add_value().data().reshape(r0.shape);
        }else if(r0.can_add_directly()){
            r0.overwrite_or_add_value().data().reshape(m_sr);
            // r0 += dot(p0,p1)
            cuv::prod(r0.overwrite_or_add_value().data(),
                    vp0,
                    vp1,
                    m_p0t, m_p1t,
                    1.f,1.f);
            r0.overwrite_or_add_value().data().reshape(r0.shape);
        }else{
            // allocate new value *sigh*
            value_ptr v(new value_type(m_sr, value_ptr::s_allocator));
            cuv::prod(*v, 
                    vp0,
                    vp1,
                    m_p0t, m_p1t);
            v->reshape(r0.shape);
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
            value_type delta = r0.delta.cdata();
            delta.reshape(m_sr);
            value_type p1v   = p1.value.cdata();
            p1v.reshape(m_sp1);
            char p1t    = (m_p0t==m_p1t) ? 't':'n';
            char deltat = (m_p0t=='t')   ? 't':'n';
            if(p0.can_overwrite_directly()){
                p0.overwrite_or_add_value().data().reshape(m_sp0);
                if(m_p0t=='n')
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            delta, p1v, deltat, p1t,1.f,0.f);
                else
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            p1v, delta, p1t, deltat,1.f,0.f);
                p0.overwrite_or_add_value().data().reshape(p0.shape);
            }
            else if(p0.can_add_directly()){
                p0.overwrite_or_add_value().data().reshape(m_sp0);
                if(m_p0t=='n')
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            delta, p1v, deltat, p1t,1.f,1.f);
                else
                    cuv::prod(p0.overwrite_or_add_value().data(),
                            p1v, delta, p1t, deltat,1.f,1.f);
                p0.overwrite_or_add_value().data().reshape(p0.shape);
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(m_sp0, value_ptr::s_allocator));
                if(m_p0t=='n')
                    cuv::prod(v.data(), delta, p1v,
                            deltat, p1t,1.f,0.f);
                else
                    cuv::prod(v.data(), p1v, delta,
                            p1t, deltat,1.f,0.f);
                v->reshape(p0.shape);
                p0.push(v);
            }
        }
        if(p1.need_derivative){
            value_type delta = r0.delta.cdata();
            value_type p0v   = p0.value.cdata();
            delta.reshape(m_sr);
            p0v.reshape(m_sp0);

            char p0t    = (m_p0t==m_p1t) ? 't':'n';
            char deltat = (m_p1t=='t')   ? 't':'n';
            if(p1.can_overwrite_directly()){
                p1.overwrite_or_add_value().data().reshape(m_sp1);
                if(m_p1t=='n')
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            p0v, delta, p0t, deltat,1.f,0.f);
                else
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            delta,p0v, deltat, p0t,1.f,0.f);
                p1.overwrite_or_add_value().data().reshape(p1.shape);
            }
            else if(p1.can_add_directly()){
                p1.overwrite_or_add_value().data().reshape(m_sp1);
                if(m_p1t=='n')
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            p0v, delta, p0t,deltat,1.f,1.f);
                else
                    cuv::prod(p1.overwrite_or_add_value().data(),
                            delta,p0v, deltat, p0t,1.f,1.f);
                p1.overwrite_or_add_value().data().reshape(p1.shape);
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(m_sp1, value_ptr::s_allocator));
                if(m_p1t=='n')
                    cuv::prod(v.data(),
                            p0v, delta, p0t,deltat,1.f,0.f);
                else
                    cuv::prod(v.data(),
                            delta,p0v, deltat,p0t,1.f,0.f);
                v->reshape(p1.shape);
                p1.push(v);
            }
        }
        r0.delta.reset();
    }

    void Prod::_determine_shapes(){
        param_t&  p0 = m_params[0];
        param_t&  p1 = m_params[1];

        m_sp0.resize(2);
        m_sp1.resize(2);
        m_sr.resize(2);
        unsigned int size_p0 = std::accumulate(p0->shape.begin(), p0->shape.end(), 1, std::multiplies<unsigned int>());
        unsigned int size_p1 = std::accumulate(p1->shape.begin(), p1->shape.end(), 1, std::multiplies<unsigned int>());
        if(m_p0t == 't'){
           m_sp0[0] = p0->shape[0];
           m_sp0[1] = size_p0 / p0->shape[0];
        }else{
           m_sp0[0] = size_p0 / p0->shape.back();
           m_sp0[1] = p0->shape.back();
        }
        if(m_p1t == 'n'){
           m_sp1[0] = p1->shape[0];
           m_sp1[1] = size_p1 / p1->shape[0];
        }else{
           m_sp1[0] = size_p1 / p1->shape.back();
           m_sp1[1] = p1->shape.back();
        }

        unsigned int n = m_p0t=='n' ? m_sp0[0] : m_sp0[1];
        unsigned int m = m_p1t=='n' ? m_sp1[1] : m_sp1[0];

        unsigned int k0 = m_p0t=='n' ? m_sp0[1] : m_sp0[0];
        unsigned int k1 = m_p1t=='n' ? m_sp1[0] : m_sp1[1];
        m_sr[0] = n;
        m_sr[1] = m;

        cuvAssert(k0==k1);
        m_results[0]->shape = p0->shape;
        if(m_p0t == 't')
            std::reverse(m_results[0]->shape.begin(), m_results[0]->shape.end());
        m_results[0]->shape.pop_back();
        std::vector<unsigned int> tmp = p1->shape;
        if(m_p1t == 't')
            std::reverse(tmp.begin(), tmp.end());
        std::copy(tmp.begin()+1, tmp.end(), std::back_inserter(m_results[0]->shape));
        
        log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("determine_shapes"));
        LOG4CXX_WARN(log, "matrix product ("<<n<<"x"<<k0<<") * ("
                <<k1<<"x"<<m<<") --> "
                <<"("<<n<<"x"<<m<<")");
    }

}
