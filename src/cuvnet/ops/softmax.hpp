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
                        value_type& result = r0.overwrite_or_add_value().data();
                        apply_binary_functor(result, inp0, inp1, BF_LOGCE_OF_LOGISTIC);
                    }else{
                        value_ptr presult  = p0.value;
                        value_type& result = presult.data_onlyshape();
                        apply_binary_functor(result, inp0, inp1, BF_LOGCE_OF_LOGISTIC);
                        r0.push(presult);
                    }

                    if(!p0.need_derivative && !p1.need_derivative) {
                        p0.value.reset();
                        p1.value.reset();
                    }
                    else if(!p1.need_derivative && p0.need_derivative)
                        p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative || p1.need_derivative);

                    if(p0.need_derivative){
                        // TODO: calculating the derivative of
                        // NegCrossEntropyOfLogistic w.r.t. param0 is quite
                        // slow, implement separate functor in CUV if needed

                        // f(x)   := 1/(1+exp(-x))
                        // L(x,z) := x*log(f(z)) + (1-x)*log(1-f(z));
                        // 0 == diff(-L(x,y),x) - (logaddexp(0,-y)-logaddexp(0,y));

                        // try to overwrite p1
                        value_ptr v = p1.value;
                        if(!p1.need_derivative)
                            p1.value.reset();

                        const value_type& p1orig = v.cdata();
                        value_type   l1(p1.shape);
                        value_type&  l2  = v.data_onlyshape();
                        cuv::apply_scalar_functor(l1,  p1orig, SF_LOGADDEXP, 0.f);
                        cuv::apply_scalar_functor(l2, -p1orig, SF_LOGADDEXP, 0.f);
                        l2 -= l1;
                        l2 *= r0.delta.cdata();
                        p0.push(v);
                    }
                    if(p1.need_derivative){
                        // f(x)   := 1/(1+exp(-x))
                        // L(x,z) := x*log(f(z)) + (1-x)*log(1-f(z));
                        // 0 == diff(-L(x,y),y) - (f(y)-x);
                        
                        // p1.delta = r0.delta * (logistic(Y)-X) 
                        if(p1.can_overwrite_directly()){
                            value_type& res = p1.overwrite_or_add_value().data();
                            apply_scalar_functor(
                                    res,
                                    p1.value.cdata(),
                                    SF_SIGM);
                            res -= p0.value.cdata();
                            res *= r0.delta.cdata();
                        }else if(p1.can_add_directly()){
                            value_type& res = p1.overwrite_or_add_value().data();
                            const value_type& p1orig = p1.value.cdata();
                            // overwrite p1
                            apply_scalar_functor(*p1.value, p1orig, SF_SIGM);
                            *p1.value -= p0.value.cdata();
                            *p1.value *= r0.delta.cdata();
                            res       += *p1.value;
                        }else{
                            const value_type& p1orig = p1.value.cdata();
                            // overwrite p1
                            apply_scalar_functor(*p1.value, p1orig, SF_SIGM);
                            *p1.value -= p0.value.cdata();
                            *p1.value *= r0.delta.cdata();
                            p1.push(p1.value);
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
