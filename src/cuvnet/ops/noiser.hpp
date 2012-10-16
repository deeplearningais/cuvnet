#ifndef __OP_NOISER_HPP__
#     define __OP_NOISER_HPP__
#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * add gaussian noise to inputs or zero them out.
     *
     * This Op may be used to 
     * - add gaussian noise of given standard deviation to the input
     * - set a given percentage of the input to zero
     *
     * bprop works correctly for both.
     *
     * @ingroup Ops
     */
    class Noiser
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
                enum NoiseType{ NT_NORMAL, NT_ZERO_OUT };
            private:
                float m_param;
                NoiseType m_noisetype;
                bool m_active;
                
                cuv::tensor<unsigned char,value_type::memory_space_type> m_zero_mask;

            public:
                Noiser(){} /// for serialization
                Noiser(result_t& p0, float param, NoiseType noise_type=NT_NORMAL)
                    :Op(1,1), m_param(param), m_noisetype(noise_type)
                     {
                         add_param(0,p0);
                     }

                /**
                 * turn noise on/off.
                 * @param b whether to turn noise on or off
                 */
                inline void set_active(bool b=true){ m_active = b; }

                /// @overload
                void release_data();

            private:
                /** 
                 * set some values to zero.
                 */
                void fprop_zero_out();
                void fprop_normal();

            public:
                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_param & m_noisetype & m_active;
                    }
        };
}
#endif /* __OP_NOISER_HPP__ */
