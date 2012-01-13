#ifndef __OP_SOFTMAX_HPP__
#     define __OP_SOFTMAX_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * calculates negative cross-entropy of logistic (pointwise)
     * 
     * \f$- x \log z - (1-x) \log(1-z)\f$, where \f$z = 1/(1-\exp(-y))\f$
     */
    class NegCrossEntropyOfLogistic
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            public:
                NegCrossEntropyOfLogistic(){} /// for serialization
                NegCrossEntropyOfLogistic(result_t& p0, result_t& p1)
                    :Op(2,1){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp0 = p0.value.cdata();           // original
                    const value_type& inp1 = p1.value.cdata();           // original

                    if(r0.can_overwrite_directly()){
                        apply_binary_functor(r0.overwrite_or_add_value().data(), inp0, inp1, BF_LOGCE_OF_LOGISTIC);
                    }else{
                        value_ptr presult  = p0.value;
                        value_type& result = presult.data_onlyshape();
                        apply_binary_functor(result, inp0, inp1, BF_LOGCE_OF_LOGISTIC);
                        r0.push(presult);
                    }
                    if(!p0.need_derivative) p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative || p1.need_derivative);

                    value_ptr delta_orig = r0.delta;
                    if(p0.need_derivative){
                        throw std::runtime_error("cannot derive NegCrossEntropyOfLogistic w.r.t. first parameter!");
                        //     log(logistic(-Y)) - log(logistic(Y)) (checked in maxima)
                        // ==  log1p(1-Y)-log1p(Y) 
                    }
                    if(p1.need_derivative){
                        // r0.delta = logistic(Y)-X (checked in maxima)
                        if(p1.can_add_directly()){
                            // p1 += r0.delta * p0.value
                            value_type v(p1.shape);
                            cuv::apply_binary_functor(v,
                                    delta_orig.cdata(),
                                    p0.value.cdata(), BF_MULT);
                            p1.overwrite_or_add_value().data() += v;
                        }else if(p1.can_overwrite_directly()){
                            // p1  := r0.delta * p0.value
                            cuv::apply_binary_functor(p1.overwrite_or_add_value().data(),
                                    delta_orig.cdata(),
                                    p0.value.cdata(),
                                    BF_MULT);
                        }else{
                            // try to overwrite delta_orig
                            const value_type& inp = delta_orig.cdata();
                            value_type& outp      = delta_orig.data_onlyshape();
                            cuv::apply_binary_functor(
                                    outp, inp, p0.value.cdata(), BF_MULT);
                            p1.push(delta_orig);
                        }
                    }
                    p0.value.reset();
                    p1.value.reset();
                    r0.delta.reset();
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
#endif /* __OP_SOFTMAX_HPP__ */
