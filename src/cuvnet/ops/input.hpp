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
                value_ptr m_data;

            public:
                Input(){} /// for serialization
                template<class T>
                    Input(const T& init):Op(0,1), m_data(new value_type(init)){  }
                void fprop(){
                    m_results[0]->push(m_data);
                    // TODO: forget m_data now?
                }
                void bprop(){}
                void _determine_shapes(){
                    m_results[0]->shape = m_data->shape();
                }
                inline value_ptr&        data_ptr()     { return m_data; }
                inline const value_ptr&  data_ptr()const{ return m_data; }

                inline value_type&       data()      { return m_data.data();  }
                inline const value_type& data() const{ return m_data.cdata(); }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_data;
                    }
        };


	
}

#endif /* __OP_INPUT_HPP__ */
