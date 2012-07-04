#ifndef __OP_MAT_PLUS_VEC_HPP__
#     define __OP_MAT_PLUS_VEC_HPP__

#include <cuvnet/op.hpp>
namespace cuvnet
{
    /**
     * adds a vector to a tensor.
     *
     * Depending on the axis parameter, the vector must have the same length as
     * the first or last dimension of the tensor.
     *
     * @ingroup Ops
     */
    class MatPlusVec
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                unsigned int m_axis;
            public:
                MatPlusVec():Op(2,1){} // for serialization
                
                /**
                 * @param mat the matrix 
                 * @param vec the vector
                 * @param axis the axis-th dimension of matrix must agree with the vector dimension.
                 *
                 * @warning currently, only the first and the last dimension of matrix are supported.
                 */
                MatPlusVec(result_t& mat, result_t& vec, unsigned int axis)
                    :   Op(2,1)
                      , m_axis(axis)
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
                        if(m_axis!=0)
                            cuv::matrix_plus_row(*r0.overwrite_or_add_value(), p1.value.cdata());
                        else
                            cuv::matrix_plus_col(*r0.overwrite_or_add_value(), p1.value.cdata());
                    }
                    else if(r0.can_add_directly()){
                        *r0.overwrite_or_add_value() += p0.value.cdata();
                        if(m_axis!=0)
                            cuv::matrix_plus_row(*r0.overwrite_or_add_value(), p1.value.cdata());
                        else
                            cuv::matrix_plus_col(*r0.overwrite_or_add_value(), p1.value.cdata());
                    }else{
                        // reallocate *sigh*
                        value_ptr v = p0.value;
                        p0.value.reset(); // try to overwrite p0
                        if(m_axis!=0) cuv::matrix_plus_row(*v, p1.value.cdata());
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
                            if(m_axis!=0)
                                reduce_to_row(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                            else
                                reduce_to_col(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                        }
                        else if(p1.can_add_directly()){
                            if(m_axis!=0)
                                reduce_to_row(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 1.f);
                            else
                                reduce_to_col(*p1.overwrite_or_add_value(),
                                        r0.delta.cdata(),RF_ADD, 1.f, 1.f);
                        }else{
                            // reallocate *sigh*
                            value_ptr v(new value_type(p1.shape));
                            if(m_axis!=0)
                                reduce_to_row(*v, r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                            else
                                reduce_to_col(*v, r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                            p1.push(v);
                        }
                    }
                    r0.delta.reset();
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape.size()>=2);
                    assert(m_params[1]->shape.size()==1);
                    assert(m_axis == 0 || m_axis == m_params[0]->shape.size()-1);
                    assert(m_params[0]->shape[m_axis] == m_params[1]->shape[0]);
                    m_results[0]->shape = m_params[0]->shape;
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
     * Multiply a matrix with a vector.
     *
     * Depending on the axis parameter, the vector must have the same length as
     * the first or last dimension of the tensor.
     *
     * @ingroup Ops
     */
    class MatTimesVec
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                unsigned int m_axis;
            public:
                MatTimesVec():Op(2,1){} // for serialization
                MatTimesVec(result_t& mat, result_t& vec, unsigned int axis)
                    :   Op(2,1)
                      , m_axis(axis)
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
                        r0.overwrite_or_add_value() = p0.value; // will be copied when written to in next lines
                        if(!p0.need_derivative)
                            p0.value.reset();   // avoid copy if it is not needed anymore
                        if(m_axis!=0)
                            cuv::matrix_times_row(*r0.overwrite_or_add_value(), p1.value.cdata());
                        else
                            cuv::matrix_times_col(*r0.overwrite_or_add_value(), p1.value.cdata());
                    }
                    else if(r0.can_add_directly()){
                        *r0.overwrite_or_add_value() += p0.value.cdata();
                        if(m_axis!=0)
                            cuv::matrix_times_row(*r0.overwrite_or_add_value(), p1.value.cdata());
                        else
                            cuv::matrix_times_col(*r0.overwrite_or_add_value(), p1.value.cdata());
                    }else{
                        // reallocate *sigh*
                        value_ptr v = p0.value; // will be copied when written to in next lines
                        if(!p1.need_derivative)
                            p0.value.reset();   // avoid copy if it is not needed anymore
                        if(m_axis!=0) cuv::matrix_times_row(*v, p1.value.cdata());
                        else          cuv::matrix_times_col(*v, p1.value.cdata());
                        r0.push(v);
                    }
                    if(!p1.need_derivative)
                        p0.value.reset();
                    if(!p0.need_derivative)
                        p1.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    if(p0.need_derivative){
                        // p0.delta = matrix_times_{row/col} ( r.delta, v )
                        if(!p1.need_derivative){
                            // try overwriting r0.delta
                            value_ptr& v = r0.delta;
                            if(m_axis!=0) matrix_times_row(v.data(),p1.value.cdata());
                            else          matrix_times_col(v.data(),p1.value.cdata());
                            p0.push(v);
                        }else{
                            // cannot overwrite r0, we need it later
                            value_ptr  v(new value_type(r0.delta.cdata().copy()));
                            if(m_axis!=0) matrix_times_row(v.data(),p1.value.cdata());
                            else          matrix_times_col(v.data(),p1.value.cdata());
                            p0.push(v);
                        }
                    }
                    if(p1.need_derivative){
                        // try overwriting r0.delta
                        const value_type& r0delta = r0.delta.cdata(); // remember true value of r0.delta
                        value_type& m = r0.delta.data_onlyshape();    // this /may/ be the same as r0delta
                        apply_binary_functor(m, r0delta, p0.value.cdata(), BF_MULT);

                        // p0.delta = reduce_to_{row/col} ( r.delta * m )
                        if(p1.can_overwrite_directly()){
                            if(m_axis!=0)
                                reduce_to_row(*p1.overwrite_or_add_value(),
                                        m,RF_ADD, 1.f, 0.f);
                            else
                                reduce_to_col(*p1.overwrite_or_add_value(),
                                        m,RF_ADD, 1.f, 0.f);
                        }
                        else if(p1.can_add_directly()){
                            if(m_axis!=0)
                                reduce_to_row(*p1.overwrite_or_add_value(),
                                        m,RF_ADD, 1.f, 1.f);
                            else
                                reduce_to_col(*p1.overwrite_or_add_value(),
                                        m,RF_ADD, 1.f, 1.f);
                        }else{
                            // reallocate *sigh*
                            value_ptr v(new value_type(p1.shape));
                            if(m_axis!=0)
                                reduce_to_row(*v, m,RF_ADD, 1.f, 0.f);
                            else
                                reduce_to_col(*v, m,RF_ADD, 1.f, 0.f);
                            p1.push(v);
                        }
                    }
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape.size()>=2);
                    assert(m_params[1]->shape.size()==1);
                    assert(m_axis == 0  || m_axis == m_params[0]->shape.size()-1);
                    assert(m_params[0]->shape[m_axis] == m_params[1]->shape[0]);
                    m_results[0]->shape = m_params[0]->shape;
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
#endif /* __OP_MAT_PLUS_VEC_HPP__ */
