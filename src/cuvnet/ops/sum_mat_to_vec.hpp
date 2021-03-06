#ifndef __OP_SUM_MAT_TO_VEC_HPP__
#     define __OP_SUM_MAT_TO_VEC_HPP__

#include <cuvnet/op.hpp>
#include <numeric>
namespace cuvnet
{
    /**
     * Sum all entries up to the specified dimension.
     *
     * E.g. a (3,4,5)-sized input summed to dimension 0 results in a
     * 1-dimensional vector with size (3).
     *
     * @ingroup Ops
     *
     */
    class SumMatToVec
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                unsigned int m_axis;
                bool m_identity;
                float m_n_summed;
                bool m_mean;
                bool m_squared;
            public:
                SumMatToVec() :   Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param mat the n-dimensional array
                 * @param axis currently, either 0 or n-1 is allowed
                 * @param mean if true, divide by the number of entries summed over
                 * @param squared if true, sum squared elements
                 */
                SumMatToVec(result_t& mat, unsigned int axis, bool mean=false, bool squared=false)
                    :   Op(1,1)
                      , m_axis(axis)
                      , m_identity(false)
                      , m_mean(mean)
                      , m_squared(squared)
            {
                add_param(0,mat);
            }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                void fprop();
                void bprop();
                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                        ar & m_mean;
                        ar & m_identity;
                        ar & m_squared;
                    }
    };

#if 0
    /**
     * Sum out one axis.
     *
     * E.g. a (3,4,5)-sized input with summed out dimension 0 results in a
     * 2-dimensional vector with size (4,5).
     *
     * Due to limitations in cuv, only first and last axis are supported.
     *
     * @ingroup Ops
     *
     */
    class SumOutAxis
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                unsigned int m_axis;
                bool m_identity;
            public:
                SumMatToVec() :   Op(1,1){} // for serialization
                SumMatToVec(result_t& mat, unsigned int axis)
                    :   Op(1,1)
                      , m_axis(axis)
                      , m_identity(false)
            {
                add_param(0,mat);
            }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    if(m_identity){
                        desc.label = "reduce to vec (optimized out)";
                        return;
                    }
                    if(m_axis == 0)
                        desc.label = "reduce->col";
                    else if(m_axis == 1)
                        desc.label = "reduce->row";
                    else 
                        desc.label = "reduce->" + boost::lexical_cast<std::string>(m_axis);
                }
                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    if(m_identity){
                        r0.push(p0.value);
                        p0.value.reset();
                        return;
                    }
                    if(r0.can_overwrite_directly()){
                        if(m_axis!=0) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata());
                        else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata());
                    }
                    else if(r0.can_add_directly()){
                        if(m_axis!=0) cuv::reduce_to_row(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD,1.f,1.f);
                        else          cuv::reduce_to_col(*r0.overwrite_or_add_value(), p0.value.cdata(),RF_ADD,1.f,1.f);
                    }else{
                        // reallocate *sigh*
                        value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                        if(m_axis!=0) cuv::reduce_to_row(*v, p0.value.cdata());
                        else          cuv::reduce_to_col(*v, p0.value.cdata());
                        r0.push(v);
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    if(m_identity){
                        p0.push(r0.delta);
                        r0.delta.reset();
                        return;
                    }

                    assert(p0.need_derivative);
                    if(p0.can_overwrite_directly()){
                        // TODO: cuv: add factOld, factNew to matrix_plus_col, matrix_times_col!!
                        p0.overwrite_or_add_value().data() = 0.f;
                        if(m_axis!=0) matrix_plus_row(*p0.overwrite_or_add_value(), r0.delta.cdata());
                        else          matrix_plus_col(*p0.overwrite_or_add_value(), r0.delta.cdata());
                    }else if(p0.can_add_directly()){
                        if(m_axis!=0) matrix_plus_row(*p0.overwrite_or_add_value(), r0.delta.cdata());
                        else          matrix_plus_col(*p0.overwrite_or_add_value(), r0.delta.cdata());
                    }else{
                        value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
                        *v = 0.f;
                        if(m_axis!=0) matrix_plus_row(*v, r0.delta.cdata());
                        else          matrix_plus_col(*v, r0.delta.cdata());
                        p0.push(v);
                    }
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape.size()>=2);
                    assert(m_axis == 0 || m_axis == m_params[0]->shape.size()-1);
                    unsigned int all
                        = std::accumulate(
                                m_params[0]->shape.begin(),
                                m_params[0]->shape.end(),
                                1,std::multiplies<unsigned int>());
                    m_results[0]->shape = std::vector<unsigned int>(1,m_params[0]->shape[m_axis]);
                    if(all / m_params[0]->shape[m_axis] == 1)
                        m_identity = true;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                        ar & m_identity;
                    }
    };
#endif
}
#endif /* __OP_SUM_MAT_TO_VEC_HPP__ */
