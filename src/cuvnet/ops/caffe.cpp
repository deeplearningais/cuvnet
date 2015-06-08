
#include <cuvnet/ops/caffe.hpp>

namespace cuvnet
{

/***************************************************
 * ResponseNormalizationAcrossMapsCaffe
 ***************************************************/

void ResponseNormalizationAcrossMapsCaffe::release_data(){
    m_denom.dealloc();
    m_orig_out.reset();
    Op::release_data();
}

void ResponseNormalizationAcrossMapsCaffe::fprop(){
	 using namespace cuv::caffe;

    param_t::element_type&  p0 = *m_params[0];
    result_t::element_type& r0 = *m_results[0];

    int nImages = p0.shape[0];
    int nMaps = p0.shape[1];
    int height = p0.shape[2];
    int width = p0.shape[3];

    m_denom.resize(r0.shape);
    if(r0.can_overwrite_directly()){
        // note: we need to /first/ run the function, /then/ copy the cow_ptr!
        //       otherwise only a copy will be overwritten.
    	local_response_normalization_across_maps((p0.value.cdata()).ptr(), m_denom.ptr(), nImages, nMaps, height, width, m_group_size, m_add_scale , m_pow_scale, (*r0.overwrite_or_add_value()).ptr());
        m_orig_out = r0.overwrite_or_add_value();
    }else{
       value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
       local_response_normalization_across_maps((p0.value.cdata()).ptr(), m_denom.ptr(), nImages, nMaps, height, width, m_group_size, m_add_scale , m_pow_scale, (*v).ptr());
        r0.push(v);
        m_orig_out = v;

    }
    if(!p0.need_derivative) {
        p0.value.reset();
        m_denom.dealloc();
        m_orig_out.reset();
    }
}

void ResponseNormalizationAcrossMapsCaffe::bprop(){
	using namespace cuv::caffe;

    param_t::element_type&  p0 = *m_params[0];
    result_t::element_type& r0 = *m_results[0];

    int nImages = p0.shape[0];
    int nMaps = p0.shape[1];
    int height = p0.shape[2];
    int width = p0.shape[3];

    if(p0.can_overwrite_directly()){
    	local_response_normalization_across_maps_grad(p0.value.cdata().ptr(), (*m_orig_out).ptr(), m_denom.ptr(), r0.delta.cdata().ptr(), nImages, nMaps, height,
        		width, m_group_size, m_add_scale, m_pow_scale,(*p0.overwrite_or_add_value()).ptr());

    }else if(p0.can_add_directly()){
    	local_response_normalization_across_maps_grad(p0.value.cdata().ptr(), (*m_orig_out).ptr(), m_denom.ptr(), r0.delta.cdata().ptr(),nImages, nMaps, height,
        		width, m_group_size, m_add_scale, m_pow_scale,(*p0.overwrite_or_add_value()).ptr(), 1.f, 1.f);
    }else{
        value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));

        local_response_normalization_across_maps_grad(p0.value.cdata().ptr(),(*m_orig_out).ptr(), m_denom.ptr(), r0.delta.cdata().ptr(),nImages, nMaps, height,
        		width, m_group_size, m_add_scale, m_pow_scale, (*v).ptr());
        p0.push(v);
    }
    r0.delta.reset();
    m_orig_out.reset();
    m_denom.dealloc();
}


void ResponseNormalizationAcrossMapsCaffe::_determine_shapes(){
    /*
     * images    (numImages, numFilters, imgPixY, imgPixX)
     * dst:      (numImages, numFilters, imgPixY, imgPixX)
     */

    assert(m_params[0]->shape.size()==4);
    if(m_group_size <= 0)
        m_group_size = m_params[0]->shape[1];
    m_results[0]->shape = m_params[0]->shape;
}




}
