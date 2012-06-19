#ifndef __OP_ATAN2_HPP__
#     define __OP_ATAN2_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * calculates pointwise atan2(Y, X), where X, Y are tensors
     *
     * @ingroup Ops
     */
    class Atan2
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            public:
                Atan2(){} /// for serialization
                /**
                 * @param p0 y
                 * @param p1 x
                 */
                Atan2(result_t& p0, result_t& p1)
                    :Op(2,1){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& y = p0.value.cdata();           // original
                    const value_type& x = p1.value.cdata();           // original

                    if(r0.can_overwrite_directly()){
                        apply_binary_functor(r0.overwrite_or_add_value().data(), y, x, BF_ATAN2);
                    }else{
                        value_ptr presult  = p0.value;
                        value_type& result = presult.data_onlyshape();
                        apply_binary_functor(result, y, x, BF_ATAN2);
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
                    assert(p0.need_derivative || p1.need_derivative);

                    value_ptr delta_orig;
                    if(p1.need_derivative) 
                        delta_orig = r0.delta;

                    const value_type& y = p0.value.cdata();           // original
                    const value_type& x = p1.value.cdata();           // original
                    
                    // determine denom = x^2 + y^2
                    value_type denom(y.shape());
                    {
                        value_type y_sq(y.shape());
                        cuv::apply_scalar_functor(denom, x, cuv::SF_SQUARE); // x^2
                        cuv::apply_scalar_functor(y_sq,  y, cuv::SF_SQUARE); // y^2
                        denom += y_sq;                                       // x^2+y^2
                        // need to add scalar here to be more robust? (INV does that already...)
                        //denom += 0.001f;
                        cuv::apply_scalar_functor(denom, cuv::SF_INV);       // 1/(x^2+y^2)
                    }

                    if(p0.need_derivative){ // derivativ w.r.t. y
                        // p0.delta := r0.delta * -(x/(x*x + y*y))
                        if(p0.can_overwrite_directly()){
                            value_type& oav = p0.overwrite_or_add_value().data();
                            cuv::apply_binary_functor(oav,
                                    x, denom,  // x / (x^2+y^2)
                                    BF_MULT);
                            oav *= r0.delta.cdata();            // r0.delta * x / (x^2+y^2)
                        }else{
                            // try to overwrite r0->delta
                            const value_type& r0delta = r0.delta.cdata();
                            value_type& outp          = r0.delta.data_onlyshape();
                            cuv::apply_binary_functor( outp,
                                    x, r0delta, BF_MULT);
                            outp *= denom;
                            p0.push(r0.delta);
                        }
                    }
                    if( p1.need_derivative){
                        // p1.delta := r0.delta * (y/(x*x + y*y))
                        if(p1.can_overwrite_directly()){
                            value_type& oav = p1.overwrite_or_add_value().data();
                            cuv::apply_binary_functor(oav,
                                    y, denom,
                                    BF_MULT);
                            oav *= -1.f;       
                            oav *= delta_orig.cdata();
                        }else{
                            // try to overwrite r0->delta
                            r0.delta.reset();
                            const value_type& r0delta = delta_orig.cdata();
                            value_type& outp          = delta_orig.data_onlyshape();
                            cuv::apply_binary_functor( outp,
                                    r0delta, y, BF_MULT);
                            outp *= -1.f;
                            outp *= denom;
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

#endif /* __OP_ATAN2_HPP__ */
