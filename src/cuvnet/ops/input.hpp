#ifndef __OP_INPUT_HPP__
#     define __OP_INPUT_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    class Input
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                value_ptr   m_data;
                std::string m_name;
                bool        m_derivable; ///< for testing ops which cannot derive w.r.t. some parameter

            public:
                Input(){} /// for serialization
                template<class T>
                    Input(const T& init):Op(0,1), m_data(new value_type(init)),m_derivable(true){  }
                template<class T>
                    Input(const T& init, const std::string& name):Op(0,1), m_data(new value_type(init)), m_name(name),m_derivable(true){  }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    desc.label = "Input `" + m_name + "'";
                }
                void fprop(){
                    m_results[0]->push(m_data);
                    // TODO: forget m_data now?
                }
                void bprop(){}
                void _determine_shapes(){
                    assert(m_data->size()>0);
                    m_results[0]->shape = m_data->shape();
                }
                inline value_ptr&        data_ptr()     { return m_data; }
                inline const value_ptr&  data_ptr()const{ return m_data; }

                inline value_type&       data()      { return m_data.data();  }
                inline const value_type& data() const{ return m_data.cdata(); }

                inline std::string&       name()      { return m_name; }
                inline const std::string& name() const{ return m_name; }

                inline bool     derivable()const{return m_derivable;}
                inline void set_derivable(bool b){m_derivable = b;}
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_data & m_name;
                    }
        };


	
}

#endif /* __OP_INPUT_HPP__ */
