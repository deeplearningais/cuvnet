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
     * The optional second output of the class is its own gradient, 
     * i.e.  \f$y_i = 1 \text{ if $x_i>0$, else 0}\f$
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
                RectifiedLinear():m_mem_optimized(false){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 input to apply rectification to
                 */
                RectifiedLinear(result_t& p0, bool mem_optimized=false):
                    Op(1,2),
                    m_result(cuvnet::get_global_allocator()),
                    m_mem_optimized(mem_optimized){
                    add_param(0,p0);
                }

                void fprop();
                void bprop();
                void _determine_shape();
            private:
                bool m_mem_optimized;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        if(version > 0)
                            ar & m_mem_optimized;
                    }
        };
}
BOOST_CLASS_VERSION(cuvnet::RectifiedLinear, 1)
#endif /* __RECTIFIED_LINEAR_HPP__ */
