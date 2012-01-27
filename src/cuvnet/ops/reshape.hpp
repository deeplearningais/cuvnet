#ifndef __OP_RESHAPE_HPP__
#     define __OP_RESHAPE_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Similar to Reshape(), but the shape is inferred from the shape of the
     * parameter.
     * 
     * \b Parameters:
     * - First parameter is the value that needs reshaping,
     * - Second (optional) parameter is the number of dimensions in the returned variable.
     *
     * \b Returns a variable with the same shape as x in the leading outdim-1
     * dimensions, but with all remaining dimensions of x collapsed into the
     * last dimension.
     *
     * For \b example, if we flatten a tensor of shape (2,3,4,5) with flatten(x,
     * outdim=2), then we’ll have the same (2-1=1) leading dimensions (2,), and
     * the remaining dimensions are collapsed. So the output in this example
     * would have shape (2, 60).
     * 
     */
    class Flatten
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_outdim;
            public:
                Flatten(result_t& in, unsigned int outdim=1)
                    :Op(1,1),
                    m_outdim(outdim)
                {
                    add_param(0,in);
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    if(r0.can_overwrite_directly()){
                        value_type& v = *r0.overwrite_or_add_value();
                        v = p0.value.cdata(); // this is O(1), but violates const-correctness(!)
                        v.reshape(r0.shape);
                    }else{
                        value_ptr v(new value_type(p0.value.cdata())); // this is O(1), but violates const-correctness(!)
                        v->reshape(r0.shape);
                        r0.push(v);
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    if(p0.can_overwrite_directly()){
                        value_type& v = *p0.overwrite_or_add_value();
                        v = r0.delta.cdata(); // O(1), but violates const-correctness again!
                        v.reshape(p0.shape);
                    }else{
                        value_ptr v(new value_type(p0.shape));
                        *v = r0.delta.cdata(); // O(1), but violates const-correctness
                        v->reshape(p0.shape);
                        p0.push(v);
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
                    assert(m_params[0]->shape.size() >= m_outdim);
                    std::vector<unsigned int> p0 = m_params[0]->shape;
                    std::vector<unsigned int> dst(m_outdim);
                    for(unsigned int i=0;i<m_outdim-1;i++)
                        dst[i] = p0[i];
                    unsigned int size = 1;
                    for(unsigned int i=m_outdim-1;i<p0.size();i++)
                        size *= p0[i];
                    dst[m_outdim-1] = size;
                    m_results[0]->shape = dst;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_outdim;
                    }
        };

    /**
     * Ensures that output has certain shape.
     *
     * one component can be <0, which means that the shape there is deduced
     * from the input dimensions.
     * 
     */
    class Reshape
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                std::vector<int> m_shape;
            public:
                template<std::size_t D>
                Reshape(result_t& in, const cuv::extent_gen<D>& eg)
                    :Op(1,1),
                    m_shape(D)
                {
                    add_param(0,in);
                    for(unsigned int i=0; i<D; i++){
                        m_shape[i] = eg.ranges_[i].finish();
                    }
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    if(r0.can_overwrite_directly()){
                        value_type& v = *r0.overwrite_or_add_value();
                        v = p0.value.cdata(); // this is O(1), but violates const-correctness(!)
                        v.reshape(r0.shape);
                    }else{
                        value_ptr v(new value_type(p0.value.cdata())); // this is O(1), but violates const-correctness(!)
                        v->reshape(r0.shape);
                        r0.push(v);
                    }
                    p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    if(p0.can_overwrite_directly()){
                        value_type& v = *p0.overwrite_or_add_value();
                        v = r0.delta.cdata(); // O(1), but violates const-correctness again!
                        v.reshape(p0.shape);
                    }else{
                        value_ptr v(new value_type(p0.shape));
                        *v = r0.delta.cdata(); // O(1), but violates const-correctness
                        v->reshape(p0.shape);
                        p0.push(v);
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
                    std::vector<unsigned int> p0 = m_params[0]->shape;

                    int special = 0;
                    for (unsigned int i = 0; i < m_shape.size(); ++i){
                        cuvAssert(m_shape[i]!=0);
                        special += m_shape[i]<0;
                    }
                    if(!special){
                        // no negative values
                        m_results[0]->shape.clear();
                        m_results[0]->shape.reserve(m_shape.size());
                        std::copy(m_shape.begin(), m_shape.end(), std::back_inserter(m_results[0]->shape));
                        return;
                    }
                    cuvAssert(special==1); // only works if /one/ dimension must be deduced
                    std::vector<unsigned int> dst(m_shape.size());
                    unsigned int n_in  =  std::accumulate(p0.begin(),p0.end(),1,std::multiplies<unsigned int>());
                    unsigned int n_out = -std::accumulate(m_shape.begin(),m_shape.end(),1,std::multiplies<int>());
                    cuvAssert(n_in%n_out == 0);
                    for (unsigned int i = 0; i < m_shape.size(); ++i){
                        if(m_shape[i]>0) dst[i] = m_shape[i];
                        else             dst[i] = n_in/n_out;
                    }
                    
                    m_results[0]->shape = dst;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_shape;
                    }
        };
}

#endif /* __OP_RESHAPE_HPP__ */
