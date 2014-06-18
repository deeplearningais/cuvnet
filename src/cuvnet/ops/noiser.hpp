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
                /// the two noise types supported by Noiser
                enum NoiseType{ 
                    NT_NORMAL,  ///< add Gaussian noise
                    NT_ZERO_OUT,  ///< set a certain percentage to zero
                    NT_SALT_AND_PEPPER  ///< set certain percentage to either 0 or 1
                };
            private:
                float m_param;
                NoiseType m_noisetype;
                bool m_active;
                bool m_compensate;
                
                cuv::tensor<unsigned char,value_type::memory_space_type> m_zero_mask;

            public:
                Noiser(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 the data to apply noise to
                 * @param param controls amount of noise
                 * @param noise_type type of noise to apply
                 * @param compensate if true and noise_type==NT_ZERO_OUT,
                 *        compensate for units set to zero by amplifying remaining
                 *        ones
                 */
                Noiser(result_t& p0, float param, NoiseType noise_type=NT_NORMAL, bool compensate=true)
                    :Op(1,1), m_param(param), m_noisetype(noise_type), m_active(true), m_compensate(compensate)
                     {
                         add_param(0,p0);
                     }

                /**
                 * set amount of noise.
                 * @param f how much noise to apply.
                 */
                inline void set_param(float f){ m_param = f; }

                /**
                 * tell how much noise is applied
                 * @return how much noise is applied
                 */
                inline float get_param()const{ return m_param; }

                /**
                 * turn noise on/off.
                 * @param b whether to turn noise on or off
                 */
                inline void set_active(bool b=true){ m_active = b; }
                
                /// @overload
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;


                /**
                 * tell whether the noiser is currently active.
                 * @return whether the noiser is currently active.
                 */
                inline bool is_active()const{ return m_active; }

                /// @overload
                void release_data();

            private:
                /** 
                 * set some values to zero.
                 */
                void fprop_zero_out();
                /**
                 * set some values to zero, others to 1
                 */
                void fprop_salt_and_pepper();
                /**
                 * adds gaussian noise.
                 */
                void fprop_normal();

            public:
                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_param & m_noisetype & m_active & m_compensate;
                    }
        };
}
#endif /* __OP_NOISER_HPP__ */
