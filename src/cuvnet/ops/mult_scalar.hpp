#ifndef __MULT_SCALAR_HPP__
#     define __MULT_SCALAR_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * fprop calculates  \f$ f(x) = \alpha x\f$
     *
     * for some scalar \f$\alpha\f$ and some input tensor \f$x\f$
     *
     * @ingroup Ops
     * 
     */
    class MultScalar
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                float m_scalar;
            public:
                /// default ctor, for serialization only.
                MultScalar()
                    :   Op(1,1)
                        , m_scalar(0)
            {
            }
                MultScalar(result_t& mat, float f)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }
                MultScalar(float f, result_t& mat)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
    };
}
#endif /* __MULT_SCALAR_HPP__ */
