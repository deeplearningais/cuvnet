#ifndef __OP_ADD_SCALAR_HPP__
#     define __OP_ADD_SCALAR_HPP__

#include <cuvnet/op.hpp>
namespace cuvnet
{
    class AddScalar
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                float m_scalar;
            public:
                AddScalar(result_t& mat, float f)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }
                AddScalar(float f, result_t& mat)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    // TODO cuv: a = a + b*scalar
                    if(r0.can_overwrite_directly()){
                        apply_scalar_functor(r0.overwrite_or_add_value().data(),p0.value.cdata(),SF_ADD,m_scalar);
                    }
                    else if(r0.can_add_directly()){
                        r0.overwrite_or_add_value().data()+=p0.value.cdata();
                        r0.overwrite_or_add_value().data()+=m_scalar;
                    }else{
                        // reallocate *sigh*
                        value_ptr v = p0.value;
                        p0.value.reset(); // try to overwrite p0
                        *v += m_scalar;
                        r0.push(v);
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    assert(p0.need_derivative);
                    p0.push(r0.delta);
                    r0.delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
    };

    class SubtractFromScalar
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                float m_scalar;
            public:
                SubtractFromScalar() :   Op(1,1){}
                SubtractFromScalar(result_t& mat, float f)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }
                SubtractFromScalar(float f, result_t& mat)
                    :   Op(1,1)
                        , m_scalar(f)
            {
                add_param(0,mat);
            }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    // TODO cuv: a = a + b*scalar SAXPY
                    if(r0.can_overwrite_directly()){
                        apply_scalar_functor(r0.overwrite_or_add_value().data(),p0.value.cdata(),SF_RSUB,m_scalar);
                    }
                    else if(r0.can_add_directly()){
                        r0.overwrite_or_add_value().data()+=m_scalar;
                        r0.overwrite_or_add_value().data()-=p0.value.cdata();
                    }else{
                        // reallocate *sigh*
                        value_ptr v = p0.value;
                        p0.value.reset(); // try to overwrite p0
                        apply_scalar_functor(*v,SF_NEGATE); // SAXPY!
                        *v += m_scalar;
                        r0.push(v);
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    assert(p0.need_derivative);
                    if(p0.can_overwrite_directly()){
                        apply_scalar_functor(p0.overwrite_or_add_value().data(),r0.delta.cdata(),SF_NEGATE);
                    }else if(p0.can_add_directly()){
                        apply_scalar_functor(r0.delta.data(),SF_NEGATE);
                        p0.overwrite_or_add_value().data()+=r0.delta.cdata(); // SAXPY!
                    }else{
                        apply_scalar_functor(r0.delta.data(),SF_NEGATE);
                        p0.push(r0.delta);
                    }
                    r0.delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
    };
}
#endif /* __OP_ADD_SCALAR_HPP__ */
