#ifndef __OP_MAT_PLUS_VEC_HPP__
#     define __OP_MAT_PLUS_VEC_HPP__

#include <cuvnet/op.hpp>
namespace cuvnet
{
    class MatPlusVec
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                bool m_row_vec;
            public:
                MatPlusVec(result_t& mat, result_t& vec, bool row)
                    :   Op(2,1)
                      , m_row_vec(row)
            {
                add_param(0,mat);
                add_param(1,vec);
            }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    if(r0.can_overwrite_directly()){
                        r0.overwrite_or_add_value() = p0.value;
                        p0.value.reset(); // try to overwrite p0
                        if(m_row_vec)
                            cuv::matrix_plus_row(*r0.overwrite_or_add_value(), p1.value.cdata());
                        else
                            cuv::matrix_plus_col(*r0.overwrite_or_add_value(), p1.value.cdata());
                    }
                    else if(r0.can_add_directly()){
                        *r0.overwrite_or_add_value() += p0.value.cdata();
                        if(m_row_vec)
                            cuv::matrix_plus_row(*r0.overwrite_or_add_value(), p1.value.cdata());
                        else
                            cuv::matrix_plus_col(*r0.overwrite_or_add_value(), p1.value.cdata());
                    }else{
                        // reallocate *sigh*
                        value_ptr v = p0.value;
                        p0.value.reset(); // try to overwrite p0
                        if(m_row_vec) cuv::matrix_plus_row(*v, p1.value.cdata());
                        else          cuv::matrix_plus_col(*v, p1.value.cdata());
                        r0.push(v);
                    }
                    p0.value.reset();
                    p1.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    if(p0.need_derivative)
                        p0.push(r0.delta);
                    if(p1.need_derivative){
                        if(p1.can_overwrite_directly()){
                            if(m_row_vec)
                                reduce_to_row(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                            else
                                reduce_to_col(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                        }
                        else if(p1.can_add_directly()){
                            if(m_row_vec)
                                reduce_to_row(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 1.f);
                            else
                                reduce_to_col(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 1.f);
                        }else{
                            // reallocate *sigh*
                            value_ptr v(new value_type(p1.shape));
                            if(m_row_vec)
                                reduce_to_row(*v, r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                            else
                                reduce_to_col(*v, r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                            p1.push(v);
                        }
                    }
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape.size()==2);
                    assert(m_params[1]->shape.size()==1);
                    if(m_row_vec) {
                        assert(m_params[0]->shape[1] == m_params[1]->shape[0]);
                    }else{
                        assert(m_params[0]->shape[0] == m_params[1]->shape[0]);
                    }
                    m_results[0]->shape = m_params[0]->shape;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_row_vec;
                    }
    };
}
#endif /* __OP_MAT_PLUS_VEC_HPP__ */
