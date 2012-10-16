#ifndef __OP_PROD_HPP__
#     define __OP_PROD_HPP__

#include <cuvnet/op.hpp>
namespace cuvnet
{
    /**
     * Matrix product.
     *
     * @ingroup Ops
     */
    class Prod
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                char m_p0t, m_p1t;
            public:
                Prod(){} /// for serialization
                Prod(result_t& p0, result_t& p1, char p0t='n', char p1t='n')
                    :Op(2,1)
                    ,m_p0t(p0t)
                    ,m_p1t(p1t)
                {
                    add_param(0,p0);
                    add_param(1,p1);
                }

                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_p0t & m_p1t;
                    }
        };
}
#endif /* __OP_PROD_HPP__ */
