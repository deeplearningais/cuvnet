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
                    m_results[0]->shape = std::vector<unsigned int>(1,1);
                }
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
                float m_thresh;

            public:
                F2Measure():Op(2,5){} /// for serialization
                F2Measure(result_t& teacher, result_t& result, float thresh = 0.f)
                    :Op(2,5)
                    ,m_thresh(thresh)
                {
                    add_param(0,teacher);
                    add_param(1,result);
                }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    desc.label = "F2Measure";
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];

                    const value_type& tch = p0.value.cdata();           // original
                    const value_type& res = p1.value.cdata();           // original

                    cuv::tensor<unsigned char, Op::value_type::memory_space_type> vtch (tch.shape());
                    cuv::tensor<unsigned char, Op::value_type::memory_space_type> vres (tch.shape());

                    vres = res > m_thresh;
                    vtch = tch > m_thresh;
                    float tp = cuv::count( vres &&  vtch, (unsigned char)1);
                    float tn = cuv::count( vres ||  vtch, (unsigned char)0);
                    float fp = cuv::count( vres && !vtch, (unsigned char)1);
                    float fn = res.size() - (tp+tn+fp);

                    float precision = tp / (tp + fp);
                    float recall    = tp / (tp + fn);
                    float f2 = 2 * precision * recall / ( precision + recall );
                    if(m_results[0]->can_overwrite_directly()){
                        m_results[0]->overwrite_or_add_value().data() = f2;
                    }else{
                        value_ptr t_f2( new cuv::tensor<float, cuv::dev_memory_space> (cuv::extents[1]) );
                        (*t_f2)[0] = f2;
                        m_results[0]->push(t_f2);
                    }
                    if(1 || m_results[1]->need_result){
                        value_ptr t_tp( new cuv::tensor<float, cuv::dev_memory_space> (cuv::extents[1]) );
                        value_ptr t_tn( new cuv::tensor<float, cuv::dev_memory_space> (cuv::extents[1]) );
                        value_ptr t_fp( new cuv::tensor<float, cuv::dev_memory_space> (cuv::extents[1]) );
                        value_ptr t_fn( new cuv::tensor<float, cuv::dev_memory_space> (cuv::extents[1]) );
                        (*t_tp)[0] = tp;
                        (*t_tn)[0] = tn;
                        (*t_fp)[0] = fp;
                        (*t_fn)[0] = fn;

                        m_results[1]->push(t_tp);
                        m_results[2]->push(t_tn);
                        m_results[3]->push(t_fp);
                        m_results[4]->push(t_fn);
                    }

                    p0.value.reset(); // forget it
                    p1.value.reset(); // forget it
                }
                void bprop(){
                    throw std::runtime_error("there is no derivative for the zero-one loss!");
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape == m_params[1]->shape);
                    m_results[0]->shape = std::vector<unsigned int>(1,1);
                    m_results[1]->shape = std::vector<unsigned int>(1,1);
                    m_results[2]->shape = std::vector<unsigned int>(1,1);
                    m_results[3]->shape = std::vector<unsigned int>(1,1);
                    m_results[4]->shape = std::vector<unsigned int>(1,1);
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_thresh;
                    }
        };
    
}
#endif /* __CLASSIFICATION_ERROR_HPP__ */
