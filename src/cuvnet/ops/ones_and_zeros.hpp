#ifndef __ONES_AND_ZEROS__
#     define __ONES_AND_ZEROS__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * creates an array of scalars the same shape as its input.
     *
     * This acts like a multi-dimensional constant.
     *
     * \ingroup Ops
     */
    class ScalarLike
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                ScalarLike(){} ///< for serialization
                
                /**
                 * ctor.
                 * @param p0 input to get shape from
                 * @param scalar constant value of the created block
                 */
                ScalarLike(result_t& p0, float scalar):Op(1,2), m_scalar(scalar){
                    add_param(0,p0);
                }

                void fprop();
                void bprop();
            private:
                float m_scalar;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
        };
}
#endif /* __ONES_AND_ZEROS__ */
