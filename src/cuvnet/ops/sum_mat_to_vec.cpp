#include "sum_mat_to_vec.hpp"

namespace cuvnet
{
    void SumMatToVec::_graphviz_node_desc(detail::graphviz_node& desc)const{
        if(m_identity){
            desc.label = "reduce to vec (optimized out)";
            return;
        }
        if(m_axis == 0)
            desc.label = "reduce->col";
        else if(m_axis == 1)
            desc.label = "reduce->row";
        else 
            desc.label = "reduce->" + boost::lexical_cast<std::string>(m_axis);
        if(m_mean){
            desc.label += " (mean)";
        }
        if(m_squared){
            desc.label += " (squared)";
        }
    }

    void SumMatToVec::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(m_identity){
            r0.push(p0.value);
            p0.value.reset();
            return;
        }
        float fact_new = m_mean ? 1.f/m_n_summed : 1.f;
        if(m_squared){
            // sum up all squared entries
            if(r0.can_overwrite_directly()){
                if(m_axis!=0) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata(), RF_ADD_SQUARED, fact_new, 0.f);
                else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata(), RF_ADD_SQUARED, fact_new, 0.f);
            }
            else if(r0.can_add_directly()){
                if(m_axis!=0) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD_SQUARED,fact_new,1.f);
                else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD_SQUARED,fact_new,1.f);
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(r0.shape));
                if(m_axis!=0) cuv::reduce_to_row(*v, p0.value.cdata(), RF_ADD_SQUARED, fact_new, 0.f);
                else          cuv::reduce_to_col(*v, p0.value.cdata(), RF_ADD_SQUARED, fact_new, 0.f);
                r0.push(v);
            }
            if(!p0.need_derivative)
                p0.value.reset(); // needed for bprop
        }else{
            // sum up all entries
            if(r0.can_overwrite_directly()){
                if(m_axis!=0) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata(), RF_ADD, fact_new, 0.f);
                else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata(), RF_ADD, fact_new, 0.f);
            }
            else if(r0.can_add_directly()){
                if(m_axis!=0) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD,fact_new,1.f);
                else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD,fact_new,1.f);
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(r0.shape));
                if(m_axis!=0) cuv::reduce_to_row(*v, p0.value.cdata(), RF_ADD, fact_new, 0.f);
                else          cuv::reduce_to_col(*v, p0.value.cdata(), RF_ADD, fact_new, 0.f);
                r0.push(v);
            }
            p0.value.reset();
        }

    }

    void SumMatToVec::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        if(m_identity){
            p0.push(r0.delta);
            r0.delta.reset();
            return;
        }

        float fact_new = m_mean ? 1.f/m_n_summed : 1.f;
        assert(p0.need_derivative);
        if(!m_squared){
            if(p0.can_overwrite_directly()){
                matrix_op_vec(
                        *p0.overwrite_or_add_value(), 
                        *p0.overwrite_or_add_value(), 
                        r0.delta.cdata(), m_axis, BF_2ND, fact_new, 0.f);
            }else if(p0.can_add_directly()){
                matrix_op_vec(
                        *p0.overwrite_or_add_value(), 
                        *p0.overwrite_or_add_value(), 
                        r0.delta.cdata(), m_axis, BF_ADD, fact_new, 1.f);
            }else{
                value_ptr v(new value_type(p0.shape));
                matrix_op_vec(
                        *v, *v, 
                        r0.delta.cdata(), m_axis, BF_2ND, fact_new, 0.f);
                p0.push(v);
            }
        }else{
            // try to overwrite p0
            const value_type& p0value = p0.value.cdata();
            value_type& dst = p0.value.data_onlyshape();

            apply_scalar_functor(dst, p0value, SF_MULT, 2.f); // ideally in-place
            matrix_op_vec(
                    dst,
                    dst,
                    r0.delta.cdata(), m_axis, BF_MULT, fact_new, 0.f);
            p0.push(p0.value);
            p0.value.reset();
        }
    }

    void SumMatToVec::_determine_shapes(){
        assert(m_params[0]->shape.size()>=2);
        assert(m_axis == 0 || m_axis == m_params[0]->shape.size()-1);
        unsigned int all
            = std::accumulate(
                    m_params[0]->shape.begin(),
                    m_params[0]->shape.end(),
                    1,std::multiplies<unsigned int>());
        m_results[0]->shape = std::vector<unsigned int>(1,m_params[0]->shape[m_axis]);

        m_n_summed = all / m_params[0]->shape[m_axis];
        if( m_n_summed == 1)
            m_identity = true;
    }
}
