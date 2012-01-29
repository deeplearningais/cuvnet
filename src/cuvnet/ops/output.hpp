#ifndef __OP_OUTPUT_HPP__
#     define __OP_OUTPUT_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    class Output
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Output(){} /// for serialization
                Output(result_t& p0):Op(1,1){ 
                    add_param(0,p0);
                }
                void fprop(){
                    // simply do not reset the m_params[0] to keep the value
                }
                void bprop(){}
                //void _determine_shapes(){ }
                //value_type&       data()      { return m_data; }
                const value_type& cdata() const{ return m_params[0]->value.cdata(); }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };

}
#endif /* __OP_OUTPUT_HPP__ */
