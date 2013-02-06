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
                std::string m_name; ///< a name for visualization
                std::vector<unsigned int> m_shape; ///< the extents of this input
                bool        m_derivable; ///< for testing ops which cannot derive w.r.t. some parameter

            public:

                Input(){} ///< for serialization

                /**
                 * ctor.
                 * @param shape the shape of the input.
                 */
                template<std::size_t D>
                Input(const cuv::extent_gen<D>& shape):Op(0,1), m_derivable(false), m_learnrate_factor(1.f), m_weight_decay_factor(1.f)
                {  
                    m_shape.resize(D);
                    for(unsigned int i=0; i<D; i++)
                        m_shape[i] = shape.ranges_[i].finish();
                }
                /**
                 * ctor.
                 * @param shape the shape of the input.
                 * @param name the name of the input.
                 */
                template<std::size_t D>
                Input(const cuv::extent_gen<D>& shape, const std::string& name):Op(0,1), m_name(name),m_derivable(false),m_learnrate_factor(1.f),m_weight_decay_factor(1.f)
                {  
                    m_shape.resize(D);
                    for(unsigned int i=0; i<D; i++)
                        m_shape[i] = shape.ranges_[i].finish();
                }
                /**
                 * ctor.
                 * @param shape the shape of the input.
                 * @param name the name of the input.
                 */
                Input(const std::vector<unsigned int>& shape, const std::string& name):Op(0,1), m_name(name),m_shape(shape),m_derivable(false),m_learnrate_factor(1.f),m_weight_decay_factor(1.f)
                {  
                }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    if(m_name.size())
                        desc.label = m_name;
                    else
                        desc.label = "Input";
                }
                virtual void fprop();
                virtual void bprop();
                void _determine_shapes();
                /**
                 * for gradient testing: this should be false for inputs for which a gradient cannot be determined.
                 */
                inline bool     derivable()const{return m_derivable;}
                /**
                 * for gradient testing: set to false for inputs for which a gradient cannot be determined.
                 * @param b whether this input can be derived for.
                 */
                inline void set_derivable(bool b){m_derivable = b;}
                /**
                 * @return the name of the Input
                 */
                inline const std::string& name()const{ return m_name; }

                /// @return a factor to multiply the learnrate with (default 1)
                inline float get_learnrate_factor(){return m_learnrate_factor;}
                /// @return a factor to multiply the weight decay with (default 1)
                inline float get_weight_decay_factor(){return m_weight_decay_factor;}
                /**
                 * set a factor to multiply learnrate with.
                 * This can be useful if different inputs should have different learning rates.
                 */
                inline void set_learnrate_factor(float learnrate_factor){m_learnrate_factor = learnrate_factor;}
                /**
                 * set a factor to multiply weight decay with.
                 * For example biases normally should not be affected by weight decay.
                 */
                inline void set_weight_decay_factor(float weight_decay_factor){m_weight_decay_factor = weight_decay_factor;}
            private:
                float m_learnrate_factor;
                float m_weight_decay_factor;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_name & m_learnrate_factor & m_shape;
                    }
        };

    /**
     * An input object wrapping a user-supplied n-dimensional data array.
     */
    class ParameterInput
        : public Input{
            public:
                typedef Op::value_type    value_type; ///< the type of the tensor
                typedef Op::op_ptr        op_ptr;     ///< a shared pointer to an Op
                typedef Op::value_ptr     value_ptr;  ///< a cow_ptr to a tensor
                typedef Op::param_t       param_t;    ///< the type of a parameter
                typedef Op::result_t      result_t;   ///< the type of a result

            private:
                value_ptr   m_data;
                value_ptr   m_delta;

            public:
                ParameterInput(){} ///< for serialization
                /**
                 * Construct a ParameterInput using eg an extents object.
                 * @param init an extents object (passed on to Input)
                 */
                template<class T>
                    ParameterInput(const T& init):Input(init), m_data(new value_type(init)){ 
                        set_derivable(true);
                    }
                /**
                 * Construct a ParameterInput using an extents object and a name
                 * @param init an extents object (passed on to Input)
                 */
                template<class T>
                    ParameterInput(const T& init, const std::string& name):Input(init,name), m_data(new value_type(init)){  
                        set_derivable(true);
                    }
                /**
                 * a useful description to be shown when visualizing a function object.
                 * @overload
                 */
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    if(m_name.size())
                        desc.label = m_name;
                    else
                        desc.label = "Input";
                }
                void fprop(); ///< @overload
                void bprop(); ///< @overload
                void _determine_shapes(); ///< @overload
                //inline void reset_delta(){ if(!!m_delta) m_delta.data()=0.f; }
                /// clear the backpropagated gradient.
                inline void reset_delta(){ m_delta.reset(); }
                /// @return the cow_ptr to the contained data
                inline value_ptr&        data_ptr()     { return m_data; }
                /// @return the cow_ptr to the contained data (const version)
                inline const value_ptr&  data_ptr()const{ return m_data; }

                /// @return the contained tensor for modification
                inline value_type&       data()      { return m_data.data();  }
                /// @return the contained tensor for read-only access
                inline const value_type& data() const{ return m_data.cdata(); }

                /// @return the backpropagated gradient
                inline       value_type& delta()      { return m_delta.data(); }
                /// @return the backpropagated gradient (read only)
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
