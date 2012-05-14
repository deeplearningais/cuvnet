#ifndef __OP_TANH_HPP__
#     define __OP_TANH_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    class Tanh
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Tanh(){} /// for serialization
                Tanh(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop(){
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
                void bprop(){
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
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
    class Sin
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Sin(){} /// for serialization
                Sin(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop(){
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
                void bprop(){
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
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
    class Cos
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Cos(){} /// for serialization
                Cos(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop(){
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
                void bprop(){
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
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
    class Logistic
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                bool m_bprop_identity;

            public:
                Logistic():m_bprop_identity(false){} /// for serialization
                Logistic(result_t& p0, bool bprop_identity=false):Op(1,1),m_bprop_identity(bprop_identity){
                    add_param(0,p0);
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();           // original
                    
                    if(r0.can_overwrite_directly()){
                        value_type& oav = p0.overwrite_or_add_value();
                        apply_scalar_functor( oav, inp, SF_SIGM);
                    }else{
                        value_type& outp = p0.value.data_onlyshape();  // if detached, only allocate same size storage
                        apply_scalar_functor( outp, inp, SF_SIGM);
                        r0.push(p0.value);      // 'copy' a newly created matrix
                    }

                    if(!p0.need_derivative)
                        p0.value.reset(); // forget it
                }
                void bprop(){
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
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}
#endif /* __OP_TANH_HPP__ */
