#include "tanh.hpp"

namespace cuvnet
{
    void Tanh::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();           // original
        value_type&      outp = p0.value.data_onlyshape();  // if detached, only allocate same size storage

        apply_scalar_functor( outp, inp, SF_TANH);

        r0.push(p0.value);      // 'copy' a newly created matrix
        if(!p0.need_derivative)
            p0.value.reset(); // forget it
    }

    void Tanh::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        const value_type& delta = r0.delta.cdata(); // this is the error from above

        const value_type& out = p0.value.cdata(); // this is the value we changed in fprop
        value_type& res       = p0.value.data_onlyshape(); // try to overwrite this

        apply_scalar_functor(res,out,SF_DTANH);
        res  *=  delta;
        p0.push(p0.value);
        p0.value.reset();
        r0.delta.reset();
    }





    void Sin::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();           // original

        if(r0.can_overwrite_directly()){
            apply_scalar_functor( r0.overwrite_or_add_value().data(), inp, SF_SIN);
        }else{
            value_ptr        outp = p0.value;
            value_type&      out  = outp.data_onlyshape();  // if detached, only allocate same size storage
            apply_scalar_functor( out, inp, SF_SIN);
            r0.push(outp);      // 'copy' a newly created matrix
        }

        if(!p0.need_derivative)
            p0.value.reset(); // forget it
    }

    void Sin::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        const value_type& delta = r0.delta.cdata(); // this is the error from above

        const value_type& inp = p0.value.cdata(); // this is the value we changed in fprop
        if(p0.can_overwrite_directly()){
            value_type& oav = p0.overwrite_or_add_value();
            apply_scalar_functor(oav,inp,SF_COS);
            oav  *=  delta;
        }else{
            value_type& res       = p0.value.data_onlyshape(); // try to overwrite this
            apply_scalar_functor(res,inp,SF_COS);
            res  *=  delta;
            p0.push(p0.value);
        }
        p0.value.reset();
        r0.delta.reset();
    }

    void Cos::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();           // original

        if(r0.can_overwrite_directly()){
            apply_scalar_functor( r0.overwrite_or_add_value().data(), inp, SF_COS);
        }else{
            value_ptr        outp = p0.value;
            value_type&      out  = outp.data_onlyshape();  // if detached, only allocate same size storage
            apply_scalar_functor( out, inp, SF_COS);
            r0.push(outp);      // 'copy' a newly created matrix
        }

        if(!p0.need_derivative)
            p0.value.reset(); // forget it
    }

    void Cos::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        const value_type& delta = r0.delta.cdata(); // this is the error from above

        const value_type& inp = p0.value.cdata(); // this is the value we changed in fprop
        if(p0.can_overwrite_directly()){
            value_type& oav = p0.overwrite_or_add_value();
            apply_scalar_functor(oav,inp,SF_SIN);
            oav  *= -1.f;
            oav  *=  delta;
        }else{
            value_type& res       = p0.value.data_onlyshape(); // try to overwrite this
            apply_scalar_functor(res,inp,SF_SIN);
            res  *= -1.f;
            res  *=  delta;
            p0.push(p0.value);
        }
        p0.value.reset();
        r0.delta.reset();
    }





    void Logistic::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();           // original

        if(!p0.need_derivative && r0.can_overwrite_directly()){
            // if we need p0.derivative, we must overwrite p0.value, since we need it for bprop
            // --> go to else{}
            value_type& oav = r0.overwrite_or_add_value();
            apply_scalar_functor( oav, inp, SF_SIGM);
        }else{
            value_type& outp = p0.value.data_onlyshape();  // if detached, only allocate same size storage
            apply_scalar_functor( outp, inp, SF_SIGM);
            r0.push(p0.value);      // 'copy' a newly created matrix
        }

        if(!p0.need_derivative)
            p0.value.reset(); // forget it
    }

    void Logistic::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        const value_type& delta = r0.delta.cdata(); // this is the error from above
        if(m_bprop_identity){
            p0.push(r0.delta);
        }else{
            const value_type& out = p0.value.cdata(); // this is the value we changed in fprop

            if(p0.can_overwrite_directly()){
                value_type& oav = p0.overwrite_or_add_value();
                apply_scalar_functor(oav,out,SF_DSIGM);
                oav *= delta;
            }else{
                value_type& res       = p0.value.data_onlyshape(); // try to overwrite this
                apply_scalar_functor(res,out,SF_DSIGM);
                res  *=  delta;
                p0.push(p0.value);
            }
            p0.value.reset();
        }
        r0.delta.reset();
    }
}
