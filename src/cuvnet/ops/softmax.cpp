#include "softmax.hpp"

namespace cuvnet
{
    void NegCrossEntropyOfLogistic::fprop(){
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

    void NegCrossEntropyOfLogistic::bprop(){
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

    void NegCrossEntropyOfLogistic::_determine_shapes(){
        assert(m_params[0]->shape == m_params[1]->shape);
        m_results[0]->shape = m_params[0]->shape;
    }



    void EpsilonInsensitiveLoss::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp0 = p0.value.cdata();           // original
        const value_type& inp1 = p1.value.cdata();           // original

        if(r0.can_overwrite_directly()){
            value_type& result = r0.overwrite_or_add_value().data();
            apply_binary_functor(result, inp0, inp1, BF_EPSILON_INSENSITIVE_LOSS, m_sensitivity);
        }else{
            value_ptr presult  = p0.value;
            value_type& result = presult.data_onlyshape();
            apply_binary_functor(result, inp0, inp1, BF_EPSILON_INSENSITIVE_LOSS, m_sensitivity);
            r0.push(presult);
        }

        if(!p0.need_derivative && !p1.need_derivative) {
            p0.value.reset();
            p1.value.reset();
        }
    }

    void EpsilonInsensitiveLoss::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        assert(!p0.need_derivative && p1.need_derivative);

        const value_type& inp0 = p0.value.cdata();           // original
        const value_type& inp1 = p1.value.cdata();           // original

        if(p1.can_overwrite_directly()){
            value_type& res = p1.overwrite_or_add_value().data();
            apply_binary_functor(res, inp0, inp1, BF_DEPSILON_INSENSITIVE_LOSS, m_sensitivity);
        }else{
            // overwrite one of the inputs
            value_ptr presult;
            if(p0.value.unique()){
                presult  = p0.value;
                p0.value.reset();
                value_type& result = presult.data_onlyshape();
                apply_binary_functor(result, inp0, inp1, BF_DEPSILON_INSENSITIVE_LOSS, m_sensitivity);
            }
            else{
                presult  = p1.value;
                p1.value.reset();
                value_type& result = presult.data_onlyshape();
                apply_binary_functor(result, inp0, inp1, BF_DEPSILON_INSENSITIVE_LOSS, m_sensitivity);
            }
            *presult *= r0.delta.cdata();
            p1.push(presult);
        }
        p0.value.reset();
        p1.value.reset();
        r0.delta.reset();
    }

    void EpsilonInsensitiveLoss::_determine_shapes(){
        assert(m_params[0]->shape == m_params[1]->shape);
        m_results[0]->shape = m_params[0]->shape;
    }




    void HingeLoss::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp0 = p0.value.cdata();           // original
        const value_type& inp1 = p1.value.cdata();           // original

        if(r0.can_overwrite_directly()){
            value_type& result = r0.overwrite_or_add_value().data();
            if(m_squared)
                apply_binary_functor(result, inp0, inp1, BF_SQHINGE_LOSS, m_margin);
            else
                apply_binary_functor(result, inp0, inp1, BF_HINGE_LOSS, m_margin);
        }else{
            value_ptr presult  = p0.value;
            value_type& result = presult.data_onlyshape();
            if(m_squared)
                apply_binary_functor(result, inp0, inp1, BF_SQHINGE_LOSS, m_margin);
            else
                apply_binary_functor(result, inp0, inp1, BF_HINGE_LOSS, m_margin);
            r0.push(presult);
        }

        if(!p0.need_derivative && !p1.need_derivative) {
            p0.value.reset();
            p1.value.reset();
        }
    }

    void HingeLoss::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        assert(!p0.need_derivative && p1.need_derivative);

        const value_type& inp0 = p0.value.cdata();           // original
        const value_type& inp1 = p1.value.cdata();           // original

        if(p1.can_overwrite_directly()){
            value_type& res = p1.overwrite_or_add_value().data();
            if(m_squared)
                apply_binary_functor(res, inp0, inp1, BF_DSQHINGE_LOSS, m_margin);
            else
                apply_binary_functor(res, inp0, inp1, BF_DHINGE_LOSS, m_margin);
        }else{
            // overwrite one of the inputs
            value_ptr presult;
            if(p0.value.unique()){
                presult  = p0.value;
                p0.value.reset();
                value_type& result = presult.data_onlyshape();
                if(m_squared)
                    apply_binary_functor(result, inp0, inp1, BF_DSQHINGE_LOSS, m_margin);
                else
                    apply_binary_functor(result, inp0, inp1, BF_DHINGE_LOSS, m_margin);
            }
            else{
                presult  = p1.value;
                p1.value.reset();
                value_type& result = presult.data_onlyshape();
                if(m_squared)
                    apply_binary_functor(result, inp0, inp1, BF_DSQHINGE_LOSS, m_margin);
                else
                    apply_binary_functor(result, inp0, inp1, BF_DHINGE_LOSS, m_margin);
            }
            *presult *= r0.delta.cdata();
            p1.push(presult);
        }
        p0.value.reset();
        p1.value.reset();
        r0.delta.reset();
    }

    void HingeLoss::_determine_shapes(){
        assert(m_params[0]->shape == m_params[1]->shape);
        m_results[0]->shape = m_params[0]->shape;
    }





    void BernoulliKullbackLeibler::fprop(){
        if(m_scalar<0.f) fprop_2p();
        else             fprop_1p();
    }

    void BernoulliKullbackLeibler::bprop(){
        if(m_scalar<0.f) bprop_2p();
        else             bprop_1p();
    }

    void BernoulliKullbackLeibler::fprop_1p(){
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

    void BernoulliKullbackLeibler::fprop_2p(){
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

    void BernoulliKullbackLeibler::bprop_1p(){
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

    void BernoulliKullbackLeibler::bprop_2p(){
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

    void BernoulliKullbackLeibler::_determine_shapes(){
        if(m_scalar<0.f){
            assert(m_params[0]->shape == m_params[1]->shape);
        }
        m_results[0]->shape = m_params[0]->shape;
    }



    void Softmax::fprop(){
        using namespace cuv;
        using namespace cuv::libs::opt;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        // reshape inputs to allow for image like dimensions
        value_type inp = p0.value.cdata();  // copy of meta info only (original data)
        std::vector<unsigned int> shape(2);
        unsigned int dim_other_axes;
        if (m_axis==0) {
            dim_other_axes = inp.size() / inp.shape(0);
            shape[0] = inp.shape(0);
            shape[1] = dim_other_axes;
        } else {
            dim_other_axes = inp.size() / inp.shape(inp.ndim()-1);
            shape[0] = dim_other_axes;
            shape[1] = inp.shape(inp.ndim()-1);
        }
        inp.reshape(shape);
        if(r0.can_overwrite_directly()){
            value_type& v = *r0.overwrite_or_add_value();
            //v.reshape(shape);
            v = 0.f; // required by cuv
            //softmax(v,p0.value.cdata(), m_axis);
            softmax(v, inp, 1-m_axis);
            v.reshape(r0.shape);
            m_result = r0.overwrite_or_add_value(); // save copy!

            // we cannot do this, we need a copy of the outputs!
            //
            //else if(r0.can_add_directly()){
            //value_type& v = *r0.overwrite_or_add_value();
            //softmax(*v,p0.value.cdata(), m_axis);
            //}

        }else{
            // try overwriting inputs
            //const value_type& src = p0.value.cdata();
            const value_type& src = inp;
            value_type& dst = p0.value.data_onlyshape();
            dst.reshape(shape);
            softmax(dst,src,1-m_axis);
            m_result = p0.value;
            //dst.reshape(r0.shape);
            r0.push(p0.value);
        }
        p0.value.reset();
    }

    void Softmax::bprop(){
        using namespace cuv;
        using namespace cuv::libs::opt;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        //value_type inp = p0.value.cdata();  // copy of meta info only (original data)
        value_type delta = r0.delta.cdata();
        std::vector<unsigned int> shape(2);
        unsigned int dim_other_axes;
         if (m_axis==0) {
            dim_other_axes = delta.size() / delta.shape(0);
            shape[0] = delta.shape(0);
            shape[1] = dim_other_axes;
        } else {
            dim_other_axes = delta.size() / delta.shape(delta.ndim()-1);
            shape[0] = dim_other_axes;
            shape[1] = delta.shape(delta.ndim()-1);
        }
        //inp.reshape(shape);
        delta.reshape(shape);
        value_type res = m_result.cdata();
        //res.reshape(shape);
        
        assert(p0.need_derivative);
        if(p0.can_overwrite_directly()){
            value_type& v = *p0.overwrite_or_add_value();
            //v.reshape(shape);
            //softmax_derivative(v,m_result.cdata(),r0.delta.cdata(), m_axis);
            softmax_derivative(v,res, delta, 1-m_axis, 0.f);
            v.reshape(r0.shape);
        }else if(0 && p0.can_add_directly()){
            // BROKEN, but is not more efficient anyway than "general" case below (softmax_derivative copies!)

            //value_type v(p0.overwrite_or_add_value()->shape()); v=0.f;
            value_type& v = *p0.overwrite_or_add_value();
            //v.reshape(shape);
            //softmax_derivative(v,m_result.cdata(),r0.delta.cdata(), m_axis);
            softmax_derivative(v,res, delta, 1-m_axis, 1.f); // 1.f: fact_old
            v.reshape(r0.shape);
            //*p0.overwrite_or_add_value() += v;
        }else{
            // try to overwrite dst
            //const value_type& delta = r0.delta.cdata();
            value_type& dst         = r0.delta.data_onlyshape();
            dst.reshape(shape);
            softmax_derivative(dst,res,delta, 1-m_axis, 0.f);
            dst.reshape(r0.shape);
            p0.push(r0.delta);
        }
        r0.delta.reset();
        m_result.reset();
    }


    

    void MultinomialLogisticLoss::fprop(){
        using namespace cuv;
        using namespace cuv::libs::opt;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        result_t::element_type& r1 = *this->result(1);
        assert(r0.need_result || r1.need_result);

        unsigned int dim_other_axes;
        value_type v0 = p0.value.cdata();  // copy of meta info only (original data)
        value_type v1 = p1.value.cdata();
        
        if (m_axis==0) {
            dim_other_axes = v0.size() / v0.shape(0);
            v0.reshape(v0.shape(0), dim_other_axes);
            v1.reshape(v1.shape(0), dim_other_axes);

        } else {
            dim_other_axes = v0.size() / v0.shape(v0.ndim()-1);
            v0.reshape(dim_other_axes, v0.shape(v0.ndim()-1));
            v1.reshape(dim_other_axes, v0.shape(v0.ndim()-1));
        }

        m_minus_logaddexp.resize(extents[dim_other_axes]);
        value_ptr ptr(new value_type(dim_other_axes, value_ptr::s_allocator));
        value_type& v = *ptr;

        if(m_axis==0){
            reduce_to_row(m_minus_logaddexp, v0, RF_LOGADDEXP,-1.f);
            reduce_to_row(v, v0 * v1, RF_ADD);
            apply_binary_functor(v,m_minus_logaddexp,BF_AXPBY, -1.f,-1.f);
        }else{
            reduce_to_col(m_minus_logaddexp, v0, RF_LOGADDEXP,-1.f);
            reduce_to_col(v, v0 * v1, RF_ADD);
            apply_binary_functor(v,m_minus_logaddexp,BF_AXPBY, -1.f,-1.f);
        }
        v.reshape(r0.shape);
        r0.push(ptr);

        if(r1.need_result){
            // calculate softmax of inputs if anybody wants them
            value_type& v0dst = p0.value.data_onlyshape();  // like data(), but does not copy
            v0dst.reshape(v0.shape(0), v0.shape(1));
            matrix_op_vec(v0dst, v0, m_minus_logaddexp, m_axis == 0 ? 1 : 0, BF_ADD);
            //apply_scalar_functor(*p0.value, SF_EXP);
            apply_scalar_functor(v0dst, SF_EXP);
            v0dst.reshape(r1.shape);
            r1.push(p0.value);
            m_softmaxed = true;
        }else
            m_softmaxed = false;
        if(!p0.need_derivative) {
            p0.value.reset();
            p1.value.reset();
        }
    }

    void MultinomialLogisticLoss::bprop(){
        using namespace cuv;
        using namespace cuv::libs::opt;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        result_t::element_type& r1 = *m_results[1];
        assert( p0.need_derivative);
        assert(!p1.need_derivative); // cannot do that currently. Why should we? :)
        assert( r0.need_result); // cannot derive for 2nd result!
        assert(!r1.need_result); // cannot derive for 2nd result!
        
        unsigned int dim_other_axes;
        value_type v0 = p0.value.cdata();
        value_type v1 = p1.value.cdata();
        if (m_axis==0) {
            dim_other_axes = v0.size() / v0.shape(0);
            v0.reshape(v0.shape(0), dim_other_axes);
            v1.reshape(v0.shape(0), dim_other_axes);
        }else{
            dim_other_axes = v0.size() / v0.shape(v0.ndim()-1);
            v0.reshape(dim_other_axes, v0.shape(v0.ndim()-1));
            v1.reshape(dim_other_axes, v0.shape(v0.ndim()-1));
        }
        value_type vd0 = r0.delta.cdata();
        vd0.reshape(cuv::extents[vd0.size()]);

        value_type& v0dst = p0.value.data_onlyshape();
        v0dst.reshape(v0.shape(0), v0.shape(1));

        value_type& v0src = v0; // remember the current source

        if(!m_softmaxed){
            // we have not calculated softmax in fprop
            matrix_op_vec(v0dst, v0src, m_minus_logaddexp, m_axis == 0 ? 1 : 0, BF_ADD);
            apply_scalar_functor(v0dst, SF_EXP);
            v0src = v0dst;
        }

        if(r1.need_result){
            value_type d1 = r1.delta.cdata();
            value_type& d1dst = r1.delta.data_onlyshape();
            d1.reshape(v0.shape(0), v0.shape(1));
            d1dst.reshape(v0.shape(0), v0.shape(1));
            
            value_type prod = v0src * d1dst;
            value_type red(v0dst.shape(m_axis == 0 ? 1 : 0));
            if(m_axis==0) reduce_to_row(red,prod,RF_ADD,-1.f);
            else          reduce_to_col(red,prod,RF_ADD,-1.f);
            
            matrix_op_vec(d1dst, d1, red, m_axis == 0 ? 1 : 0, BF_ADD);
            d1dst *= v0src;
            d1dst.reshape(r1.shape);
        }

        if(r0.need_result){
            // now bprop the above minus p1 times delta.
            v0dst -= v1;
            matrix_op_vec(v0dst, v0src, vd0, m_axis == 0 ? 1 : 0, BF_MULT);
        }

        if(r1.need_result && r0.need_result){
            //*p0.value += *r1.delta;
            v0dst += *r1.delta;
            //v0dst += d1dst;
        }else if(r1.need_result){
            //*p0.value   = *r1.delta;
            v0dst  = *r1.delta;
            //v0dst  = d1dst;
        }else if(r0.need_result){
            // :-)
        }

        v0dst.reshape(p0.shape);
        p0.push(p0.value);
        p0.value.reset();
        r0.delta.reset();
    }

    void MultinomialLogisticLoss::_determine_shapes(){
        if(m_results[0]->need_result){
            assert(m_params[0]->shape == m_params[1]->shape);
            cuvAssert(m_axis == 0 ||
                    m_axis == m_params[0]->shape.size() - 1);
        }
        std::vector<unsigned int> src = m_params[0]->shape;
        m_results[0]->shape.resize(src.size()-1);

        unsigned int k = 0;
        for(unsigned int i=0;i<src.size();i++){
            if( i!=m_axis )
                m_results[0]->shape[k++] = src[i];
        }
        m_results[1]->shape = m_params[0]->shape;
    }

    /***********************************************
     * MultinomialLogisticLoss2
     ***********************************************/

    void MultinomialLogisticLoss2::fprop(){
#ifdef CUVNET_USE_CPU
        throw std::runtime_error("MultinomialLogisticLoss2 fprop: not implemented for CPU");
#else
        using namespace cuv;
        using namespace cuv::libs::opt;
        param_t::element_type&  p0 = *m_params[0]; // estimator
        param_t::element_type&  p1 = *m_params[1]; // labels
        result_t::element_type& r0 = *m_results[0]; // loglikelihood of true labels
        result_t::element_type& r1 = *m_results[1]; // softmaxed estimator
        result_t::element_type& r2 = *m_results[2]; // classification error
        std::pair<float, float> lloss_closs;
        if(r1.can_overwrite_directly()){
            lloss_closs = cuv::libs::opt::multinomial_logistic_loss(
                    *r1.overwrite_or_add_value(), p0.value.cdata(),
                    p1.value.cdata(), m_pattern_axis, value_ptr::s_allocator);
            m_softmaxed = r1.overwrite_or_add_value();
        }else{
            // try to overwrite p0.value
            lloss_closs = cuv::libs::opt::multinomial_logistic_loss(
                    *p0.value, p0.value.cdata(),
                    p1.value.cdata(), m_pattern_axis, value_ptr::s_allocator);
            m_softmaxed = p0.value;
            r1.push(p0.value);
        }
        if(r0.need_result){
            // log loss
            if(r0.can_overwrite_directly()){
                (*r0.overwrite_or_add_value())[0] = lloss_closs.first;
            }else if(r0.can_add_directly()){
                (*r0.overwrite_or_add_value())[0] += lloss_closs.first;
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                (*v)[0] = lloss_closs.first;
                r0.push(v);
            }
        }
        if(r2.need_result){
            // classification loss
            if(r2.can_overwrite_directly()){
                (*r2.overwrite_or_add_value())[0] = lloss_closs.second;
            }else if(r2.can_add_directly()){
                (*r2.overwrite_or_add_value())[0] += lloss_closs.second;
            }else{
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                (*v)[0] = lloss_closs.second;
                r2.push(v);
            }
        }

        p0.value.reset();
        if(!p0.need_derivative){
            p1.value.reset();
            m_softmaxed.reset();
        }
#endif
    }

    void MultinomialLogisticLoss2::bprop(){
#ifdef CUVNET_USE_CPU
        throw std::runtime_error("MultinomialLogisticLoss2 bprop: not implemented for CPU");
#else
        using namespace cuv;
        using namespace cuv::libs::opt;
        param_t::element_type&  p0 = *m_params[0]; // estimator
        param_t::element_type&  p1 = *m_params[1]; // labels
        result_t::element_type& r0 = *m_results[0];
        //result_t::element_type& r1 = *m_results[1];
        //result_t::element_type& r2 = *m_results[2];
        assert( p0.need_derivative);
        assert(!p1.need_derivative); // cannot do that currently. Why should we? :)

        float f = r0.delta.cdata()[0];
        if(p0.can_overwrite_directly()){
            multinomial_logistic_loss_grad(*p0.overwrite_or_add_value(),
                    m_softmaxed.cdata(), p1.value.cdata(),
                    m_pattern_axis, f, false);
        }else if(p0.can_add_directly()){ 
            multinomial_logistic_loss_grad(*p0.overwrite_or_add_value(),
                    m_softmaxed.cdata(), p1.value.cdata(),
                    m_pattern_axis, f, true);
        }else{
            // try to overwrite the softmaxed activations
            const value_type& sm = m_softmaxed.cdata();
            multinomial_logistic_loss_grad(m_softmaxed.data_onlyshape(),
                    sm, p1.value.cdata(),
                    m_pattern_axis, f, false);
            p0.push(m_softmaxed);
        }
        m_softmaxed.reset();
#endif
    }

    void MultinomialLogisticLoss2::_determine_shapes(){
        if(m_results[0]->need_result){
            cuvAssert(m_pattern_axis == 0 ||
                    m_pattern_axis == (int)m_params[0]->shape.size() - 1);
            cuvAssert(m_params[1]->shape.size() == 1);
            cuvAssert(m_params[0]->shape[m_pattern_axis] == m_params[1]->shape[0]);
        }
        std::vector<unsigned int> src = m_params[0]->shape;
        m_results[0]->shape.resize(1); m_results[0]->shape[0] = 1;
        m_results[1]->shape = m_params[0]->shape;
        m_results[2]->shape.resize(1); m_results[2]->shape[0] = 1;
    }

}
