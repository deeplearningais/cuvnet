#include "multiply.hpp"

namespace cuvnet
{

    void Multiply::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "x * y";
    }
    void Multiply::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp0 = p0.value.cdata();           // original
        const value_type& inp1 = p1.value.cdata();           // original

        if(r0.can_overwrite_directly()){
            apply_binary_functor(r0.overwrite_or_add_value().data(), inp0, inp1, BF_MULT);
        }else{
            value_ptr presult  = p0.value;
            value_type& result = presult.data_onlyshape();
            apply_binary_functor(result, inp0, inp1, BF_MULT);
            r0.push(presult);
        }
        if(!p0.need_derivative) p1.value.reset();
        if(!p1.need_derivative) p0.value.reset();
    }
    void Multiply::bprop(){
        // TODO: CUV: implement a = a + b*c
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative || p1.need_derivative);

        value_ptr delta_orig = r0.delta;
        if(p0.need_derivative){
            if(p0.can_add_directly()){
                // p0.delta += r0.delta * p1
                value_type v(p0.shape);
                cuv::apply_binary_functor(v,
                        r0.delta.cdata(),
                        p1.value.cdata(),
                        BF_MULT);
                p0.overwrite_or_add_value().data() += v;
            }else if(p0.can_overwrite_directly()){
                // p0.delta := r0.delta * p1
                cuv::apply_binary_functor(p0.overwrite_or_add_value().data(),
                        r0.delta.cdata(),
                        p1.value.cdata(),
                        BF_MULT);
            }else{
                if(!p1.need_derivative){
                    // we can only try to overwrite the current value
                    // of r0->delta if it is not needed for p1
                    delta_orig.reset();
                }
                // try to overwrite r0->delta
                const value_type& inp = r0.delta.cdata();
                value_type& outp      = r0.delta.data_onlyshape();
                cuv::apply_binary_functor(
                        outp, p1.value.cdata(), inp, BF_MULT);
                p0.push(r0.delta);
            }
        }
        if(p1.need_derivative){
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
}
