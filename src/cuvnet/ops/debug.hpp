#ifndef __CUVNET_OP_DEBUG_HPP__
#     define __CUVNET_OP_DEBUG_HPP__
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
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

                void fprop();
                void bprop();
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
