#ifndef __RECTIFIED_LINEAR_HPP__
#     define __RECTIFIED_LINEAR_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Calculates the maximum with 0.
     *
     * i.e. \f$y_i = \max(0, x_i)\f$
     *
     * \ingroup Ops
     */
    class RectifiedLinear
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                cuv::tensor<unsigned char, matrix::memory_space_type> m_result;
            public:
                RectifiedLinear(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 input to apply rectification to
                 */
                RectifiedLinear(result_t& p0):Op(1,1){
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
#endif /* __RECTIFIED_LINEAR_HPP__ */
