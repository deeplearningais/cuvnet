#ifndef __CUVNET_OP_DEBUG_HPP__
#     define __CUVNET_OP_DEBUG_HPP__
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * prints (stats of) the values passed through to std out.
     *
     * @ingroup Ops
     */
    class Printer
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                std::string m_dbg;
                bool m_fprop, m_bprop;

            public:
                Printer(){} /// for serialization
                Printer(const std::string& name, result_t& p0, bool fprop=true, bool bprop=true)
                    : Op(1,1), m_dbg(name), m_fprop(fprop), m_bprop(bprop){
                    add_param(0,p0);
                }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    desc.label = "Printer `" + m_dbg + "'";
                }

                void fprop(){
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
                void bprop(){
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
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_dbg & m_fprop & m_bprop;
                    }
        };
}

#endif /* __CUVNET_OP_DEBUG_HPP__ */
