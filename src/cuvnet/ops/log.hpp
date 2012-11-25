#ifndef __OP_LOG_HPP__
#     define __OP_LOG_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Takes the logarithm of its inputs.
     *
     * @ingroup Ops
     */
    class Log
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Log(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 the input to take the log of
                 */
                Log(result_t& p0):Op(1,1){
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
#endif /* __OP_LOG_HPP__ */
