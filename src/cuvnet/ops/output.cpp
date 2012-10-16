#include "output.hpp"

namespace cuvnet
{
    void Sink::fprop(){

        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        if(r0.result_uses.size() > 0)
            r0.push(p0.value);

        // also, do not reset the m_params[0] to keep the value
    }

    void Sink::bprop(){}

    void Sink::forget(){
        m_params[0]->value.reset();
    }

    void Sink::_graphviz_node_desc(detail::graphviz_node& desc)const{
        if(m_name.size())
            desc.label = m_name;
        else
            desc.label = "Sink";
    }

    // does nothing!
    void DeltaSink::fprop(){}
    // simply do not reset m_result[0].delta
    void DeltaSink::bprop(){}

    void DeltaSink::forget(){
        m_results[0]->delta.reset();
    }

    void DeltaSink::_graphviz_node_desc(detail::graphviz_node& desc)const{
        if(m_name.size())
            desc.label = m_name;
        else
            desc.label = "DeltaSink";
    }


    void Pipe::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "Pipe " + boost::lexical_cast<std::string>(m_idx);
    }

    void Pipe::fprop(){
        using namespace cuv;
        param_t::element_type&  p = *m_params[0];
        result_t::element_type& r = *m_results[0];

        if(r.can_overwrite_directly()){
            r.overwrite_or_add_value().data() =p.value.cdata();
        }else if(r.can_add_directly()){
            r.overwrite_or_add_value().data()+=p.value.cdata();
        }else{
            r.push(p.value);
        }
        p.value.reset();
    }

    void Pipe::bprop(){
        using namespace cuv;
        param_t::element_type&  p = *m_params[0];
        result_t::element_type& r = *m_results[0];
        if(p.can_overwrite_directly()){
            p.overwrite_or_add_value().data() = r.delta.cdata();
        }else if(p.can_add_directly()){
            p.overwrite_or_add_value().data()+= r.delta.cdata();
        }else{
            p.push(r.delta);
        }
        r.delta.reset();
    }

}
