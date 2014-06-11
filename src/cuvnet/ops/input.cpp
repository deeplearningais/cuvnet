#include "input.hpp"

namespace cuvnet
{
    void Input::fprop(){ throw std::runtime_error("fprop() not implemented for input `"+m_name+"'!"); }
    void Input::bprop(){ throw std::runtime_error("bprop() not implemented for input `"+m_name+"'!"); }
    void Input::_determine_shapes(){
        m_results[0]->shape = m_shape;
    }

    void ParameterInput::release_data(){
        Op::release_data();
        m_delta.reset();
    }
    void ParameterInput::fprop(){
        m_results[0]->push(m_data);
        // TODO: forget m_data now? (Inputs only, not weights)
    }
    void ParameterInput::bprop(){
        if(!m_delta || m_delta.cdata().ndim()==0)
            m_delta = m_results[0]->delta;
        else 
            *m_delta += m_results[0]->delta.cdata();

        m_results[0]->delta.reset();
    }
    void ParameterInput::_determine_shapes(){
        //cuvAssert(m_data->shape() == m_shape);
        //Input::_determine_shapes();
        m_results[0]->shape = m_data->shape();
    }
}
