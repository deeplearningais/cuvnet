#ifndef __OP_OUTPUT_HPP__
#     define __OP_OUTPUT_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    class Sink
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Sink(){} /// for serialization
                Sink(const std::string& name, result_t& p0):Op(1,1),m_name(name){ 
                    add_param(0,p0);
                }
                ///  A sink always pretends it wants the result
                /// @overload
                virtual bool need_result()const{return true;}
                Sink(result_t& p0):Op(1,1){ 
                    add_param(0,p0);
                }
                void fprop(){
                    
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    if(r0.result_uses.size() > 0)
                        r0.push(p0.value);
                    
                    // also, do not reset the m_params[0] to keep the value
                }
                void bprop(){}
                //void _determine_shapes(){ }
                //value_type&       data()      { return m_data; }
                const value_type& cdata() const{ return m_params[0]->value.cdata(); }

                /**
                 * forget the stored value
                 *
                 * this may help during bprop, as it can be overwritten if only
                 * a single copy remains now.
                 */
                void forget(){
                    m_params[0]->value.reset();
                }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    desc.label = "Sink `" + m_name + "'";
                }
            private:
                std::string    m_name;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_name;
                    }
        };

    class Pipe
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
                unsigned int m_idx; /// for visualization only!

            public:
                Pipe(){} /// for serialization
                Pipe(result_t& p0, unsigned int idx):Op(1,1), m_idx(idx){ 
                    add_param(0,p0);
                }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    desc.label = "Pipe " + boost::lexical_cast<std::string>(m_idx);
                }
                void fprop(){
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
                void bprop(){
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
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_idx;
                    }
        };

}
#endif /* __OP_OUTPUT_HPP__ */
