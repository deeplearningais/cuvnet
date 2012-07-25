#ifndef __GLOBALMAXPOOL_HPP__
#     define __GLOBALMAXPOOL_HPP__

#include <cuvnet/op.hpp>
#include <boost/format.hpp>

namespace cuvnet
{
    /**
     *  sets all the elements of the rows of the matrix to zero, and element which has the maximum value in the row sets to one.  
     *
     * \f[ f_{i,j}(X) = \begin{cases}
     *                     1, & \mathrm{if\ } j = \arg \max_k(x_{i,k}) \\ 
     *                     0, & \mathrm{else}
     *                  \end{cases} \f]
     *
     *
     * @ingroup Ops
     * 
     */
    class GlobalMaxPool
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                cuv::tensor<float,cuv::host_memory_space> m_arg_max;
            public:
                /// default ctor, for serialization only.
                GlobalMaxPool()
                    :   Op(1,1)
            {
            }
                GlobalMaxPool(result_t& mat)
                    :   Op(1,1)
            {
                add_param(0,mat);
            }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    desc.label = "select_max";
                }
                
                void select_max_set_one(value_type& result, const value_type& w){
                    value_type v(cuv::extents[w.shape(0)]);
                    cuv::reduce_to_col(v, w, cuv::RF_ARGMAX);
                    //result = - 10.f; 
                    result = - 10.f; 
                    m_arg_max = v;
                    unsigned int num_rows = m_arg_max.shape(0);
                    for(unsigned int row = 0; row < num_rows; row++){
                        int col = (int)(m_arg_max(row) + 0.5f);
                        result(row, col) = w(row, col) * w(row, col) + 10.f;
                        //result(row, col) = w(row, col) + 10.f;
                    }
                }
                
                void derivative_select_max(value_type& result, const value_type& w){
                    result = 0.f; 
                    for(unsigned int row = 0; row < m_arg_max.shape(0); row++){
                        result(row, m_arg_max(row)) = 2.f * w(row, m_arg_max(row));
                        //result(row, m_arg_max(row)) = 1;
                    }
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    if(r0.can_overwrite_directly()){
                        select_max_set_one(*r0.overwrite_or_add_value(), p0.value.cdata());
                    }
                    else{
                        // reallocate *sigh*
                        value_ptr res_(new value_type(r0.shape));
                        select_max_set_one(*res_, p0.value.cdata());
                        r0.push(res_);
                    }
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    assert(p0.need_derivative);
                    if(p0.can_overwrite_directly()){
                        derivative_select_max(p0.overwrite_or_add_value().data() , p0.value.cdata());
                        p0.overwrite_or_add_value().data() *= r0.delta.cdata();
                        //apply_scalar_functor(p0.overwrite_or_add_value().data(), r0.delta.cdata(), SF_MULT, m_scalar);
                    }else{
                        //apply_binary_functor(p0.overwrite_or_add_value().data(), r0.delta.cdata(), BF_XPBY, m_scalar);
                        value_ptr temp(new value_type(p0.shape));
                        derivative_select_max(*temp , p0.value.cdata());
                        *temp *= r0.delta.cdata();
                        p0.push(temp);
                    }

                    p0.value.reset();
                    r0.delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        //ar & m_scalar;
                    }
    };
}
#endif /* __GLOBALMAXPOOL_HPP__ */
