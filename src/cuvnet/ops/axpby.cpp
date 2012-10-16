#include "axpby.hpp"
namespace cuvnet
{
    void Axpby::_graphviz_node_desc(detail::graphviz_node& desc)const{
        if( m_fact_a == 1.f && m_fact_b == 1.f)
            desc.label = "x + y";
        else if( m_fact_a == -1.f && m_fact_b == 1.f)
            desc.label = "y - x";
        else if( m_fact_a == 1.f && m_fact_b == -1.f)
            desc.label = "x - y";
        else
            desc.label = boost::str(boost::format("%2.3f x + %2.3f y")%m_fact_a%m_fact_b);
    }

    void Axpby::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp0 = p0.value.cdata();           // original
        const value_type& inp1 = p1.value.cdata();           // original
        bool write_to_p0 = p0.value.unique() && (p0.shape == r0.shape);

        if(r0.can_overwrite_directly()){
            apply_binary_functor(r0.overwrite_or_add_value().data(), inp0, inp1, BF_AXPBY, m_fact_a, m_fact_b);
        }else{
            value_type&  outp  = write_to_p0
                ? p0.value.data_onlyshape()
                : p1.value.data_onlyshape();  
            outp.resize(r0.shape);
            apply_binary_functor(outp, inp0, inp1, BF_AXPBY, m_fact_a, m_fact_b);
            r0.push(write_to_p0 ? p0.value : p1.value);
        }
        p0.value.reset(); // forget it
        p1.value.reset(); // forget it
    }

    void Axpby::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative || p1.need_derivative);

        value_ptr delta_orig = r0.delta;
        value_type::value_type s = m_scalar_param >= 0 ? sum(r0.delta.cdata()) : 0;
        if(p0.need_derivative){
            if(m_scalar_param == 0){
                if(p0.can_overwrite_directly()){
                    *p0.overwrite_or_add_value()  = s * m_fact_a;
                }else if(p0.can_add_directly()){
                    *p0.overwrite_or_add_value() += s * m_fact_a;
                }else{
                    value_ptr v(new value_type(1));
                    v.data() = s * m_fact_a;
                    p0.push(v);
                }
            }
            else if(p0.can_add_directly()){
                // p0 += fact_a * r0.delta
                cuv::apply_binary_functor(p0.overwrite_or_add_value().data(),
                        r0.delta.cdata(),
                        BF_XPBY, m_fact_a);
            }else if(p0.can_overwrite_directly()){
                // p0  = fact_a * r0.delta
                cuv::apply_scalar_functor(p0.overwrite_or_add_value().data(),
                        r0.delta.cdata(),
                        SF_MULT, m_fact_a);
            }else{
                if(!p1.need_derivative){
                    // we can only try to overwrite the current value
                    // of r0->delta if it is not needed for p1
                    delta_orig.reset();
                }
                // try to overwrite r0->delta
                const value_type& inp = r0.delta.cdata();
                value_type& outp      = r0.delta.data_onlyshape();
                cuv::apply_scalar_functor(
                        outp, inp, SF_MULT, m_fact_a);
                p0.push(r0.delta);
            }
        }
        r0.delta.reset();
        if(p1.need_derivative){
            if(m_scalar_param == 1){
                if(p1.can_overwrite_directly()){
                    *p1.overwrite_or_add_value()  = s * m_fact_b;
                }else if(p1.can_add_directly()){
                    *p1.overwrite_or_add_value() += s * m_fact_b;
                }else{
                    value_ptr v(new value_type(1));
                    v.data() = s * m_fact_b;
                    p1.push(v);
                }
            }
            else if(p1.can_add_directly()){
                // p1 += fact_b * r1.delta
                cuv::apply_binary_functor(p1.overwrite_or_add_value().data(),
                        delta_orig.cdata(),
                        BF_XPBY, m_fact_b);
            }else if(p1.can_overwrite_directly()){
                // p1  = fact_b * r1.delta
                cuv::apply_scalar_functor(p1.overwrite_or_add_value().data(),
                        delta_orig.cdata(),
                        SF_MULT, m_fact_b);
            }else{
                // try to overwrite delta_orig
                const value_type& inp = delta_orig.cdata();
                value_type& outp      = delta_orig.data_onlyshape();
                cuv::apply_scalar_functor(
                        outp, inp, SF_MULT, m_fact_b);
                p1.push(delta_orig);
            }
        }
    }

    void Axpby::_determine_shapes(){
        if(0);
        else if(m_params[0]->shape == m_params[1]->shape){
            m_results[0]->shape = m_params[0]->shape;
            m_scalar_param = -1;
        }else if(m_params[0]->shape.size()==1 && m_params[0]->shape[0]==1){
            m_results[0]->shape = m_params[1]->shape;
            m_scalar_param = 0;
        }else if(m_params[1]->shape.size()==1 && m_params[1]->shape[0]==1){
            m_results[0]->shape = m_params[0]->shape;
            m_scalar_param = 1;
        }else{
            throw std::runtime_error("AXPBY shapes do not match (must be same or one of them a scalar)");
        }
    }
}
