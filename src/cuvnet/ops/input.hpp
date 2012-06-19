#ifndef __OP_INPUT_HPP__
#     define __OP_INPUT_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * A function with zero inputs and one output.
     *
     * Change the contained tensor to your likes to use this as an input to
     * your function. For convenience, the inputs can have names, so you
     * recognize them when dumping the function using \c write_graphviz.
     *
     * Some inputs are *parameters*. That is, they represent a weight matrix or
     * a value that you want to optimize, e.g. using \c gradient_decent.
     * Such parameters are not treated differently here than inputs.
     *
     * @ingroup Ops
     */
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
                    if(m_name.size())
                        desc.label = m_name;
                    else
                        desc.label = "Input";
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
                virtual void release_data(){
                    ; // keep data!
                }
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
