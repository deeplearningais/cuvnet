#ifndef __OP_TANH_HPP__
#     define __OP_TANH_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Applies hyperbolic tangent to its inputs (elementwise).
     *
     * @ingroup Ops
     */
    class Tanh
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Tanh(){} /// for serialization
                Tanh(result_t& p0):Op(1,1){
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

    /**
     * Applies sine to its inputs (elementwise).
     *
     * @ingroup Ops
     */
    class Sin
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Sin(){} /// for serialization
                Sin(result_t& p0):Op(1,1){
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
    /**
     * Applies cosine function to all its inputs (elementwise).
     *
     * @ingroup Ops
     */
    class Cos
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Cos(){} /// for serialization
                Cos(result_t& p0):Op(1,1){
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
    
    /**
     * Applies logistic function to its inputs (elementwise).
     *
     * \f$f(x)=1/(1+\exp(-x))\f$
     *
     * @ingroup Ops
     */
    class Logistic
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                bool m_bprop_identity;

            public:
                Logistic():m_bprop_identity(false){} /// for serialization
                Logistic(result_t& p0, bool bprop_identity=false):Op(1,1),m_bprop_identity(bprop_identity){
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
#endif /* __OP_TANH_HPP__ */
