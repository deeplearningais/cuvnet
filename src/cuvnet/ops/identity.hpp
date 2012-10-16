#ifndef __OP_IDENTITY_HPP__
#     define __OP_IDENTITY_HPP__
#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * simply passes on its inputs to whoever wants it.
     *
     * This is mainly an example of how to write an \c Op, it is not really
     * useful.
     *
     * @ingroup Ops
     */
    class Identity 
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Identity(){} /// for serialization
                Identity(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}
#endif /* __OP_IDENTITY_HPP__ */
