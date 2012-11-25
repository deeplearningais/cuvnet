#ifndef __OP_SUM_HPP__
#     define __OP_SUM_HPP__

#include <cuvnet/op.hpp>
#include <iomanip>


namespace cuvnet
{
    /**
     * the Kahan stable sum algorithm (slow!).
     * @param m the values to be summed
     */
    template <class T>
        double kahan_summation(const T& m) {
            double result = 0.f;

            double c = 0.f;
            for(unsigned int i=0;i < m.size(); ++i) {
                double y = (float)m[i] - c;
                double t = result + y;
                c = (t - result) - y;
                result = t;
            }
            return result;
        }
    /**
     * Sums over all entries in its argument.
     * @ingroup Ops
     */
    class Sum
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                bool m_identity;

            public:
                Sum(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 the input to be summed
                 */
                Sum(result_t& p0):Op(1,1){
                    add_param(0,p0);
                    m_results[0]->delta           = value_ptr(new value_type(cuv::extents[1]));
                    m_results[0]->delta.data()[0] = 1.f;
                    m_identity = false;
                }
                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_identity;
                    }
        };

    /**
     * mean over all entries in its argument.
     *
     * @ingroup Ops
     */
    class Mean
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                float m_div;
                bool  m_identity;

            public:
                Mean(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 the values to be averaged
                 */
                Mean(result_t& p0):Op(1,1){
                    add_param(0,p0);
                    m_results[0]->delta           = value_ptr(new value_type(cuv::extents[1]));
                    m_results[0]->delta.data()[0] = 1.f;
                    m_identity = false;
                }
                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_identity;
                    }
        };

}
#endif /* __OP_SUM_HPP__ */
