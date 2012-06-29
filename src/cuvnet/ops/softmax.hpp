#ifndef __OP_SOFTMAX_HPP__
#     define __OP_SOFTMAX_HPP__

#include <cuvnet/op.hpp>
#include <cuv/libs/opt/opt.hpp>

namespace cuvnet
{
    /**
     * calculates negative cross-entropy of logistic (pointwise).
     * 
     * \f$- x \log z - (1-x) \log(1-z)\f$, where \f$z = 1/(1-\exp(-y))\f$
     *
     * @ingroup Ops
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
    /**
     * epsilon insensitive loss function.
     * 
     * \f$ x \max(0,1-y-a)^2 + (-x) \max(0,y-a)^2\f$.
     *
     * @note The derivative is only implemented w.r.t. \i y!
     *
     * @ingroup Ops
     */
    class EpsilonInsensitiveLoss
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
                float m_sensitivity;
            public:
                EpsilonInsensitiveLoss(){} /// for serialization
                EpsilonInsensitiveLoss(float sensitivity, result_t& p0, result_t& p1)
                    :Op(2,1), m_sensitivity(sensitivity){
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
                        apply_binary_functor(result, inp0, inp1, BF_SQSQLOSS, m_sensitivity);
                    }else{
                        value_ptr presult  = p0.value;
                        value_type& result = presult.data_onlyshape();
                        apply_binary_functor(result, inp0, inp1, BF_SQSQLOSS, m_sensitivity);
                        r0.push(presult);
                    }

                    if(!p0.need_derivative && !p1.need_derivative) {
                        p0.value.reset();
                        p1.value.reset();
                    }
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    assert(!p0.need_derivative && p1.need_derivative);

                    const value_type& inp0 = p0.value.cdata();           // original
                    const value_type& inp1 = p1.value.cdata();           // original

                    if(p1.can_overwrite_directly()){
                        value_type& res = p1.overwrite_or_add_value().data();
                        apply_binary_functor(res, inp0, inp1, BF_DSQSQLOSS, m_sensitivity);
                    }else{
                        // overwrite one of the inputs
                        value_ptr presult;
                        if(p0.value.unique()){
                            presult  = p0.value;
                            p0.value.reset();
                            value_type& result = presult.data_onlyshape();
                            apply_binary_functor(result, inp0, inp1, BF_DSQSQLOSS, m_sensitivity);
                        }
                        else{
                            presult  = p1.value;
                            p1.value.reset();
                            value_type& result = presult.data_onlyshape();
                            apply_binary_functor(result, inp0, inp1, BF_DSQSQLOSS, m_sensitivity);
                        }
                        *presult *= r0.delta.cdata();
                        p0.value.reset();
                        p1.value.reset();
                        p1.push(presult);
                    }

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
                        ar & m_sensitivity;
                    }
        };
    /**
     * calculates the elementwise KullbackLeibler-Divergence of two Bernoulli
     * variables given by their means.
     * 
     * \f$ x\log (x/y) + (1-x)\log\frac{1-x}{1-y} \f$
     *
     * @ingroup Ops
     */
    class BernoulliKullbackLeibler
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            public:
                BernoulliKullbackLeibler(){} /// for serialization
                BernoulliKullbackLeibler(result_t& p0, result_t& p1)
                    :Op(2,1), m_scalar(-1.f){
                         add_param(0,p0);
                         add_param(1,p1);
                     }
                BernoulliKullbackLeibler(float scalar, result_t& p0)
                    :Op(1,1),m_scalar(scalar){
                         add_param(0,p0);
                     }
                void fprop(){
                    if(m_scalar<0.f) fprop_2p();
                    else             fprop_1p();
                }
                void bprop(){
                    if(m_scalar<0.f) bprop_2p();
                    else             bprop_1p();
                }

                void fprop_1p(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp0 = p0.value.cdata();           // original

                    if(r0.can_overwrite_directly()){
                        value_type& result = r0.overwrite_or_add_value().data();
                        apply_scalar_functor(result, inp0, SF_BERNOULLI_KL, m_scalar);
                    }else{
                        value_ptr presult  = p0.value;
                        value_type& result = presult.data_onlyshape();
                        apply_scalar_functor(result, inp0, SF_BERNOULLI_KL, m_scalar);
                        r0.push(presult);
                    }

                    if(!p0.need_derivative) {
                        p0.value.reset();
                    }
                }
                void fprop_2p(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp0 = p0.value.cdata();           // original
                    const value_type& inp1 = p1.value.cdata();           // original

                    if(r0.can_overwrite_directly()){
                        value_type& result = r0.overwrite_or_add_value().data();
                        apply_binary_functor(result, inp0, inp1, BF_BERNOULLI_KL);
                    }else{
                        value_ptr presult  = p0.value;
                        value_type& result = presult.data_onlyshape();
                        apply_binary_functor(result, inp0, inp1, BF_BERNOULLI_KL);
                        r0.push(presult);
                    }

                    if(!p0.need_derivative && !p1.need_derivative) {
                        p0.value.reset();
                        p1.value.reset();
                    }
                }
                void bprop_1p(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative); 

                    if(p0.can_overwrite_directly()){
                        value_type& r       = p0.overwrite_or_add_value();
                        apply_scalar_functor(r,p0.value.cdata(),SF_DBERNOULLI_KL,m_scalar);
                        r *= r0.delta.cdata();
                    }else{
                        // try to overwrite p0
                        value_ptr v = p0.value;
                        p0.value.reset();
                        value_type& x       = v.data();
                        apply_scalar_functor(x,SF_DBERNOULLI_KL,m_scalar);
                        x *= r0.delta.cdata();
                        p0.push(v); 
                    }

                    p0.value.reset();
                    r0.delta.reset();
                }
                void bprop_2p(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    //assert(!p0.need_derivative && p1.need_derivative); // cannot derive for x
                    assert(p1.need_derivative); // cannot derive for x!

                    if(p0.can_overwrite_directly()){
                        value_type& r       = p0.overwrite_or_add_value();
                        const value_type& x = p0.value.cdata();
                        apply_binary_functor(r,x,p0.value.cdata(),BF_DBERNOULLI_KL);
                        r *= r0.delta.cdata();
                    }else{
                        // try to overwrite p1
                        value_ptr v = p1.value;
                        p1.value.reset();
                        value_type& y       = v .data();
                        const value_type& x = p0.value.cdata();
                        apply_binary_functor(y,x,y,BF_DBERNOULLI_KL);
                        y *= r0.delta.cdata();
                        p1.push(v); // derive w.r.t. p1!
                    }

                    p1.value.reset();
                    p0.value.reset();
                    r0.delta.reset();
                }
                void _determine_shapes(){
                    if(m_scalar<0.f){
                        assert(m_params[0]->shape == m_params[1]->shape);
                    }
                    m_results[0]->shape = m_params[0]->shape;
                }
            private:
                float m_scalar;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
        };
    /**
     * Softmax activation function.
     *
     * \f[
     * f(x_i) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}
     * \f]
     *
     * where the sum over j is either over the rows or the columns of a
     * matrix, depending on the second parameter.
     *
     * @ingroup Ops
     */
    class Softmax
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_axis;
                value_ptr m_result;
            public:
                Softmax(){} /// for serialization
                Softmax(result_t& p0, unsigned int axis)
                    :Op(1,1)
                    ,m_axis(1-axis)
                {
                    add_param(0,p0);
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::libs::opt;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    if(r0.can_overwrite_directly()){
                        value_type& v = *r0.overwrite_or_add_value();
                        v = 0.f; // required by cuv
                        softmax(v,p0.value.cdata(), m_axis);
                        m_result = r0.overwrite_or_add_value(); // save copy!
                        
                    // we cannot do this, we need a copy of the outputs!
                    //}
                    //else if(r0.can_add_directly()){
                        //value_type& v = *r0.overwrite_or_add_value();
                        //softmax(*v,p0.value.cdata(), m_axis);

                    }else{
                        // try overwriting inputs
                        const value_type& src = p0.value.cdata();
                        value_type& dst = p0.value.data_onlyshape();
                        softmax(dst,src,m_axis);
                        m_result = p0.value;
                        r0.push(p0.value);
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    using namespace cuv::libs::opt;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);
                    if(p0.can_overwrite_directly()){
                        value_type& v = *p0.overwrite_or_add_value();
                        v = 0.f;
                        softmax_derivative(v,m_result.cdata(),r0.delta.cdata(), m_axis);
                    }else if(p0.can_add_directly()){
                        value_type& v = *p0.overwrite_or_add_value();
                        softmax_derivative(v,m_result.cdata(),r0.delta.cdata(), m_axis);
                    }else{
                        // try to overwrite dst
                        const value_type& delta = r0.delta.cdata();
                        value_type& dst         = r0.delta.data_onlyshape();
                        softmax_derivative(dst,m_result.cdata(),delta, m_axis);
                        p0.push(r0.delta);
                    }
                    r0.delta.reset();
                    m_result.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };

    /**
     * Multinomial logistic loss.
     *
     * For some dataset \f$D=(X,Y)\f$
     * \f[
     * l(\theta, D) = 
     *  -\sum_{i=1}^{|X|}\left[
     *      \sum_k y_{ik}x_{ik} - \log\sum_k\exp(x_{ik})
     *     \right]
     * \f]
     *
     * where \f$x\f$ is a function of \f$\theta\f$.
     *
     * the second result of this op is the softmaxed input.
     *
     */
    class MultinomialLogisticLoss
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_axis;
                value_type m_minus_logaddexp;
                bool         m_softmaxed;
            public:
                MultinomialLogisticLoss(){} /// for serialization
                MultinomialLogisticLoss(result_t& p0, result_t& p1, unsigned int axis)
                    :Op(2,2)
                    ,m_axis(axis)
                {
                    assert(m_axis==0 || m_axis==1);
                    add_param(0,p0);
                    add_param(1,p1);
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::libs::opt;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    result_t::element_type& r1 = *this->result(1);
                    assert(r0.need_result || r1.need_result);

                    m_minus_logaddexp.resize(extents[p0.shape[1-m_axis]]);
                    value_ptr ptr(new value_type(p0.shape[1-m_axis]));
                    value_type& v = *ptr;
                    if(m_axis==0){
                        reduce_to_row(m_minus_logaddexp,p0.value.cdata(),RF_LOGADDEXP,-1.f);
                        reduce_to_row(v, p0.value.cdata()*p1.value.cdata(), RF_ADD);
                        apply_binary_functor(v,m_minus_logaddexp,BF_AXPBY, -1.f,-1.f);
                    }else if(m_axis==1){
                        reduce_to_col(m_minus_logaddexp,p0.value.cdata(),RF_LOGADDEXP,-1.f);
                        reduce_to_col(v, p0.value.cdata()*p1.value.cdata(), RF_ADD);
                        apply_binary_functor(v,m_minus_logaddexp,BF_AXPBY, -1.f,-1.f);
                    }
                    r0.push(ptr);

                    if(r1.need_result){
                        // calculate softmax of inputs if anybody wants them
                        if(m_axis==0) matrix_plus_row(*p0.value,m_minus_logaddexp);
                        else          matrix_plus_col(*p0.value,m_minus_logaddexp);
                        apply_scalar_functor(*p0.value, SF_EXP);
                        r1.push(p0.value);
                        m_softmaxed = true;
                    }else
                        m_softmaxed = false;

                    if(!p0.need_derivative) {
                        p0.value.reset();
                        p1.value.reset();
                    }
                }
                void bprop(){
                    using namespace cuv;
                    using namespace cuv::libs::opt;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    result_t::element_type& r1 = *m_results[1];
                    assert( p0.need_derivative);
                    assert(!p1.need_derivative); // cannot do that currently. Why should we? :)

                    if(!m_softmaxed){
                        // we have not calculated softmax in fprop
                        if(m_axis==0) matrix_plus_row(*p0.value,m_minus_logaddexp);
                        else          matrix_plus_col(*p0.value,m_minus_logaddexp);
                        apply_scalar_functor(*p0.value, SF_EXP);
                    }

                    if(r1.need_result){
                        value_type prod = p0.value.cdata() * r1.delta.cdata();
                        value_type red(p0.shape[1-m_axis]);
                        if(m_axis==0) reduce_to_row(red,prod,RF_ADD,-1.f);
                        else          reduce_to_col(red,prod,RF_ADD,-1.f);
                        if(m_axis==0) matrix_plus_row(*r1.delta,red);
                        else          matrix_plus_col(*r1.delta,red);
                        *r1.delta *= *p0.value;
                    }

                    if(r0.need_result){
                        // now bprop the the above minus p1 times delta.
                        *p0.value -= p1.value.cdata();
                        if(m_axis==0)
                            matrix_times_row(*p0.value, r0.delta.cdata());
                        else
                            matrix_times_col(*p0.value, r0.delta.cdata());
                    }

                    if(r1.need_result && r0.need_result){
                        *p0.value += *r1.delta;
                    }else if(r1.need_result){
                        *p0.value   = *r1.delta;
                    }else if(r0.need_result){
                        // :-)
                    }

                    p0.push(p0.value);
                    p0.value.reset();
                    r0.delta.reset();
                }
            void _determine_shapes(){
                assert(m_params[0]->shape == m_params[1]->shape);
                std::vector<unsigned int> src = m_params[0]->shape;
                m_results[0]->shape.resize(src.size()-1);

                unsigned int k = 0;
                for(unsigned int i=0;i<src.size();i++){
                    if( i!=m_axis )
                        m_results[0]->shape[k++] = src[i];
                }
                m_results[1]->shape = m_params[0]->shape;
            }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };
}
#endif /* __OP_SOFTMAX_HPP__ */
