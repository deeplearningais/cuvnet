#ifndef __OP_SUM_MAT_TO_VEC_HPP__
#     define __OP_SUM_MAT_TO_VEC_HPP__

#include <cuvnet/op.hpp>
namespace cuvnet
{
    class SumMatToVec
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
                SumMatToVec(result_t& mat, bool row)
                    :   Op(1,1)
                      , m_row_vec(row)
            {
                add_param(0,mat);
            }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    if(r0.can_overwrite_directly()){
                        if(m_row_vec) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata());
                        else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata());
                    }
                    else if(r0.can_add_directly()){
                        if(m_row_vec) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD,1.f,1.f);
                        else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD,1.f,1.f);
                    }else{
                        // reallocate *sigh*
                        value_ptr v(new value_type(r0.shape));
                        if(m_row_vec) cuv::reduce_to_row(*v, p0.value.cdata());
                        else          cuv::reduce_to_col(*v, p0.value.cdata());
                        r0.push(v);
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    // TODO: cuv: add factOld, factNew to matrix_plus_col, matrix_times_col!!
                    assert(p0.need_derivative);
                    if(p0.can_overwrite_directly()){
                        p0.overwrite_or_add_value().data() = 0.f;
                        if(m_row_vec) matrix_plus_row(*p0.overwrite_or_add_value(), r0.delta.cdata());
                        else          matrix_plus_col(*p0.overwrite_or_add_value(), r0.delta.cdata());
                    }else if(p0.can_add_directly()){
                        if(m_row_vec) matrix_plus_row(*p0.overwrite_or_add_value(), r0.delta.cdata());
                        else          matrix_plus_col(*p0.overwrite_or_add_value(), r0.delta.cdata());
                    }else{
                        value_ptr v(new value_type(p0.shape));
                        *v = 0.f;
                        if(m_row_vec) matrix_plus_row(*v, r0.delta.cdata());
                        else          matrix_plus_col(*v, r0.delta.cdata());
                        p0.push(v);
                    }
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape.size()==2);
                    if(m_row_vec)
                        m_results[0]->shape = std::vector<unsigned int>(1,m_params[0]->shape[1]);
                    else
                        m_results[0]->shape = std::vector<unsigned int>(1,m_params[0]->shape[0]);
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
#endif /* __OP_SUM_MAT_TO_VEC_HPP__ */
