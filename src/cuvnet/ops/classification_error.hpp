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
                ClassificationLoss():Op(2,1){} /// for serialization
                ClassificationLoss(result_t& p0, result_t& p1)
                    :Op(2,1){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    desc.label = "ClassificationLoss";
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp0 = p0.value.cdata();           // original
                    const value_type& inp1 = p1.value.cdata();           // original

                    int batch_size = inp0.shape(0);
                    cuv::tensor<int,Op::value_type::memory_space_type> a1 ( batch_size );
                    cuv::tensor<int,Op::value_type::memory_space_type> a2 ( batch_size );
                    cuv::reduce_to_col(a1, inp0,cuv::RF_ARGMAX);
                    cuv::reduce_to_col(a2, inp1,cuv::RF_ARGMAX);
                    
                    a1 -= a2;
                    int n_wrong = batch_size - cuv::count(a1,0);

                    value_ptr res(new value_type(cuv::extents[1]));
                    *res = n_wrong/(float)batch_size;

                    r0.push(res);
                    p0.value.reset(); // forget it
                    p1.value.reset(); // forget it
                }
                void bprop(){
                    throw std::runtime_error("there is no derivative for the zero-one loss!");
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape == m_params[1]->shape);
                    m_results[0]->shape = m_params[0]->shape;
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
    
}
#endif /* __CLASSIFICATION_ERROR_HPP__ */
