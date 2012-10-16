#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include "debug.hpp"

namespace cuvnet
{
    void Printer::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        if(m_fprop){
            log4cxx::LoggerPtr log = log4cxx::Logger::getLogger("printer");
            LOG4CXX_WARN(log, m_dbg << " forward mean: " << cuv::mean(*p0.value) << " var: " << cuv::var(*p0.value));
        }

        r0.push(p0.value);
        p0.value.reset();       // forget params
    }
    void Printer::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);
        if(m_bprop){
            log4cxx::LoggerPtr log = log4cxx::Logger::getLogger("printer");
            LOG4CXX_WARN(log, m_dbg << " backward mean: " << cuv::mean(*r0.delta) << " var: " << cuv::var(*r0.delta));
        }

        p0.push(r0.delta);
        r0.delta.reset();
    }

    void Printer::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "Printer `" + m_dbg + "'";
    }
}
