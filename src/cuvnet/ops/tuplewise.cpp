#include <cuvnet/common.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops/tuplewise.hpp>


namespace cuvnet{

void Tuplewise_op::_determine_shapes(){
    cuvAssert(m_params[0]->shape.size() > 1);
    cuvAssert(m_params[0]->shape.size() > m_dim);
    cuvAssert(m_params[0]->shape[m_dim] % m_subspace_size == 0);
    std::vector<unsigned int> dst = m_params[0]->shape;
    dst[m_dim] /= m_subspace_size;
    m_results[0]->shape = dst;
}

void Tuplewise_op::fprop(){
    using namespace cuv;
    using namespace cuv::alex_conv;
    param_t::element_type&  p0 = *m_params[0];
    result_t::element_type& r0 = *m_results[0];
    if(r0.can_overwrite_directly()){
        value_ptr& v = r0.overwrite_or_add_value();
        tuplewise_op(*v, p0.value.cdata(), m_dim, m_subspace_size, m_to, m_epsilon);
    }else{
        value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
        tuplewise_op(*v, p0.value.cdata(), m_dim, m_subspace_size, m_to, m_epsilon);
        r0.push(v);
    }
    // keep p0.value!
}

void Tuplewise_op::bprop(){
    using namespace cuv;
    using namespace cuv::alex_conv;
    param_t::element_type&  p0 = *m_params[0];
    result_t::element_type& r0 = *m_results[0];
    if(p0.can_overwrite_directly()){
        tuplewise_op_grad(*p0.overwrite_or_add_value(), p0.value.cdata(), r0.delta.cdata(), m_dim, m_subspace_size, m_to, m_epsilon);
    }else{
        const matrix& oldvalue = p0.value.cdata();
        value_type& v = p0.value.data_onlyshape();
        tuplewise_op_grad(v, oldvalue, r0.delta.cdata(), m_dim, m_subspace_size, m_to, m_epsilon);
        p0.push(p0.value);
    }
    p0.value.reset();
    r0.delta.reset();
}

void Tuplewise_op::_graphviz_node_desc(detail::graphviz_node& desc)const{
    desc.label = "Tuplewise (dim=" +
        boost::lexical_cast<std::string>(m_dim) + "/" +
        boost::lexical_cast<std::string>(m_subspace_size) + ": ";
    if(m_to == cuv::alex_conv::TO_NORM)
        desc.label += "norm";
    else if(m_to == cuv::alex_conv::TO_MAX)
        desc.label += "max";
    else if(m_to == cuv::alex_conv::TO_ADD_SQUARED)
        desc.label += "addsq";
    else
        throw std::runtime_error("unknown tuplewise op");
}

}
