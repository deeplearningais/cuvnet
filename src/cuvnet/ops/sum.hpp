#ifndef __OP_SUM_HPP__
#     define __OP_SUM_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * sums over all entries in p0
     */
    class Sum
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Sum(){} /// for serialization
                Sum(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }
                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    float sum = cuv::sum(p0.value.cdata());
                    if(r0.can_overwrite_directly()){
                        (*r0.overwrite_or_add_value())[0] = sum;
                    }
                    else if(r0.can_add_directly()){
                        (*r0.overwrite_or_add_value())[0] += sum;
                    }else{
                        // reallocate *sigh*
                        value_ptr v(new value_type(r0.shape));
                        v.data()[0] = sum;
                        r0.push(v);
                    }
                    // don't delete p0, instead overwrite it in bprop
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    if(p0.can_overwrite_directly()){
                        value_ptr& v = p0.overwrite_or_add_value();
                        v = p0.value;
                        p0.value.reset(); // try overwriting p0
                        *v = 1.f;
                    }else if(p0.can_add_directly()){
                        value_ptr& v = p0.overwrite_or_add_value();
                        *v += 1.f;
                        p0.value.reset(); // try overwriting p0
                    }else{
                        value_ptr v = p0.value; // try overwriting p0
                        p0.value.reset();
                        *v = 1.f;
                        p0.push(v);
                    }
                    //r0.delta.reset(); // do not reset delta, it is very small anyway
                }
                void _determine_shapes(){
                    m_results[0]->shape.resize(1);
                    m_results[0]->shape[0] = 1;
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };

}
#endif /* __OP_SUM_HPP__ */
