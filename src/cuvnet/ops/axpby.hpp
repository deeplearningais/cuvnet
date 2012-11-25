#ifndef __OP_AXPBY_HPP__
#     define __OP_AXPBY_HPP__

#include <cuvnet/op.hpp>
#include <boost/format.hpp>

namespace cuvnet
{
    /**
     * calculates \f$ \alpha * X + \beta * Y\f$, where
     * \f$\alpha\f$, \f$\beta\f$ are scalar values and \f$X\f$, \f$Y\f$ denote tensors.
     *
     * @ingroup Ops
     */
    class Axpby
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                float m_fact_a, m_fact_b;
                int m_scalar_param;

            public:
                Axpby():Op(2,1){} ///< for serialization

                /**
                 * ctor.
                 * \f$ \alpha * X + \beta * Y\f$
                 * @param p0 X
                 * @param p1 Y
                 * @param fact_a alpha
                 * @param fact_b beta
                 */
                Axpby(result_t& p0, result_t& p1, float fact_a=1.f, float fact_b=1.f)
                    :Op(2,1)
                     , m_fact_a(fact_a)
                     , m_fact_b(fact_b){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;

                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_fact_a & m_fact_b & m_scalar_param;
                    }
        };
}

#endif /* __OP_AXPBY_HPP__ */
