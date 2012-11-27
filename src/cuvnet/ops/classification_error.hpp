#ifndef __CLASSIFICATION_ERROR_HPP__
#     define __CLASSIFICATION_ERROR_HPP__

#include <cuvnet/op.hpp>
#include <boost/format.hpp>

namespace cuvnet
{

    /**
     * \brief calculates the classification loss for one-out-of-n codes.
     *
     * \f[ \frac{1}{N} \sum_{j=1}^N r_j = |\arg\max_i x_{ji} - \arg\max_i y_{ji}| \neq 0 \f].
     *
     * @note there is no derivative for defined for this operator. 
     *       You can use \c linear_regression or \c logistic_regression.
     *
     * @ingroup Ops
     */
    class ClassificationLoss
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:

            public:
                ClassificationLoss():Op(2,1){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 estimator
                 * @param p1 teacher
                 */
                ClassificationLoss(result_t& p0, result_t& p1)
                    :Op(2,1){
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
                    }
        };

    /**
     * \brief calculates the F2-measure for binary classification problems
     *
     * @note there is no derivative for defined for this operator. 
     *
     * @ingroup Ops
     */
    class F2Measure
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                float m_thresh_tch, m_thresh_res;

            public:
                F2Measure():Op(3,5){} ///< for serialization
                /**
                 * ctor with ignore-input.
                 * @param teacher the teacher
                 * @param result the estimator for teacher
                 * @param ignore values in estimator/teacher for which ignore is 0 are not considered.
                 * @param thresh_tch threshold for teacher
                 * @param thresh_res threshold for result
                 */
                F2Measure(result_t& teacher, result_t& result, result_t& ignore, float thresh_tch = 0.f, float thresh_res = 0.f)
                    :Op(3,5)
                    ,m_thresh_tch(thresh_tch)
                    ,m_thresh_res(thresh_res)
                {
                    add_param(0,teacher);
                    add_param(1,result);
                    add_param(2,ignore);
                }
                /**
                 * ctor.
                 * @param teacher the teacher
                 * @param result the estimator for teacher
                 * @param thresh_tch threshold for teacher
                 * @param thresh_res threshold for result
                 */
                F2Measure(result_t& teacher, result_t& result, float thresh_tch = 0.f, float thresh_res = 0.f)
                    :Op(2,5)
                    ,m_thresh_tch(thresh_tch)
                    ,m_thresh_res(thresh_res)
                {
                    add_param(0,teacher);
                    add_param(1,result);
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
                        ar & m_thresh_tch & m_thresh_res;
                    }
        };
    
}
#endif /* __CLASSIFICATION_ERROR_HPP__ */
