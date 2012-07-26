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
     * Such inputs should be instances of the more specialized \c ParameterInput.
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

            protected:
                std::string m_name;
                std::vector<unsigned int> m_shape;
                bool        m_derivable; ///< for testing ops which cannot derive w.r.t. some parameter

            public:
                float m_learnrate_factor;
                float m_weight_decay_factor;

                Input(){} /// for serialization
                template<std::size_t D>
                Input(const cuv::extent_gen<D>& shape):Op(0,1), m_derivable(false), m_learnrate_factor(1.f), m_weight_decay_factor(1.f)
                {  
                    m_shape.resize(D);
                    for(unsigned int i=0; i<D; i++)
                        m_shape[i] = shape.ranges_[i].finish();
                }
                template<std::size_t D>
                Input(const cuv::extent_gen<D>& shape, const std::string& name):Op(0,1), m_name(name),m_derivable(false),m_learnrate_factor(1.f),m_weight_decay_factor(1.f)
                {  
                    m_shape.resize(D);
                    for(unsigned int i=0; i<D; i++)
                        m_shape[i] = shape.ranges_[i].finish();
                }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    if(m_name.size())
                        desc.label = m_name;
                    else
                        desc.label = "Input";
                }
                virtual void fprop(){ throw std::runtime_error("fprop() not implemented for input `"+m_name+"'!"); }
                virtual void bprop(){ throw std::runtime_error("bprop() not implemented for input `"+m_name+"'!"); }
                void _determine_shapes(){
                    m_results[0]->shape = m_shape;
                }
                inline bool     derivable()const{return m_derivable;}
                inline void set_derivable(bool b){m_derivable = b;}
                inline const std::string& name()const{ return m_name; }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_name & m_learnrate_factor & m_shape;
                    }
        };

    class ParameterInput
        : public Input{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                value_ptr   m_data;
                value_ptr   m_delta;

            public:
                ParameterInput(){} /// for serialization
                template<class T>
                    ParameterInput(const T& init):Input(init), m_data(new value_type(init)){ 
                        set_derivable(true);
                    }
                template<class T>
                    ParameterInput(const T& init, const std::string& name):Input(init,name), m_data(new value_type(init)){  
                        set_derivable(true);
                    }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    if(m_name.size())
                        desc.label = m_name;
                    else
                        desc.label = "Input";
                }
                void fprop(){
                    m_results[0]->push(m_data);
                    // TODO: forget m_data now? (Inputs only, not weights)
                }
                void bprop(){
                    if(!m_delta || m_delta.cdata().ndim()==0)
                        m_delta = m_results[0]->delta;
                    else 
                        *m_delta += m_results[0]->delta.cdata();

                    m_results[0]->delta.reset();
                }
                void _determine_shapes(){
                    //cuvAssert(m_data->shape() == m_shape);
                    //Input::_determine_shapes();
                    m_results[0]->shape = m_data->shape();
                }
                //inline void reset_delta(){ if(!!m_delta) m_delta.data()=0.f; }
                inline void reset_delta(){ m_delta.reset(); }
                inline value_ptr&        data_ptr()     { return m_data; }
                inline const value_ptr&  data_ptr()const{ return m_data; }

                inline value_type&       data()      { return m_data.data();  }
                inline const value_type& data() const{ return m_data.cdata(); }

                inline       value_type& delta()      { return m_delta.data(); }
                inline const value_type& delta() const{ return m_delta.cdata(); }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Input>(*this);
                        ar & m_data;
                    }
        };
}

#endif /* __OP_INPUT_HPP__ */
