#include <boost/scope_exit.hpp>
#include <cuvnet/common.hpp>
#include <cuvnet/op_utils.hpp>
#include "convolve.hpp"

namespace cuvnet
{

    /***************************************************
     * SeparableFilter 1D
     ***************************************************/

    void SeparableFilter1d::fprop(){
#ifdef CUVNET_USE_CPU
        throw std::runtime_error(" fprop: not implemented for CPU");
#else
        using namespace cuv;
        using namespace cuv::libs;
        using namespace cuv::libs::nlmeans;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            if (m_dim == 0)
                convolutionRows(*r0.overwrite_or_add_value(), p0.value.cdata(), m_kernel);
            else if(m_dim == 1)
                convolutionColumns(*r0.overwrite_or_add_value(), p0.value.cdata(), m_kernel);
            else
                convolutionDepth(*r0.overwrite_or_add_value(), p0.value.cdata(), m_kernel);

        }else{
            // try to overwrite p0
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            if (m_dim == 0)
                convolutionRows(*v, p0.value.cdata(), m_kernel);
            else if(m_dim == 1)
                convolutionColumns(*v, p0.value.cdata(), m_kernel);
            else
                convolutionDepth(*v, p0.value.cdata(), m_kernel);
            r0.push(v);
        }
        p0.value.reset();
#endif
    }

    void SeparableFilter1d::bprop(){
#ifdef CUVNET_USE_CPU
        throw std::runtime_error("SeparableFilter1d bprop: not implemented for CPU");
#else
        using namespace cuv;
        using namespace cuv::libs;
        using namespace cuv::libs::nlmeans;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(p0.can_overwrite_directly()){
            if (m_dim == 0)
                convolutionRows(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_kernel_reverse);
            else if(m_dim == 1)
                convolutionColumns(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_kernel_reverse);
            else
                convolutionDepth(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_kernel_reverse);
        }else{
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            if (m_dim == 0)
                convolutionRows(*v, r0.delta.cdata(), m_kernel_reverse);
            else if(m_dim == 1)
                convolutionColumns(*v, r0.delta.cdata(), m_kernel_reverse);
            else
                convolutionDepth(*v, r0.delta.cdata(), m_kernel_reverse);

            p0.push(v);
        }
        r0.delta.reset();
        p0.value.reset();
#endif
    }

    void SeparableFilter1d::_determine_shapes(){
        /*
         * images    (numFilters, imgPixX, numImages)
         * dst:      (numFilters, imgPixX, numImages)
         */
        cuvAssert(m_params[0]->shape.size()<4);
        cuvAssert(m_kernel.shape(0) > 2);
        cuvAssert(m_kernel.ndim() == 1);
        cuvAssert(m_dim != 1); // seems to be broken in cuv
        m_results[0]->shape = m_params[0]->shape;

        unsigned int size = m_kernel.shape(0);
        m_kernel_reverse.resize(cuv::extents[size]);
        for (unsigned int i = 0; i < size; ++i)
        {
            m_kernel_reverse(i) = m_kernel(size - 1 - i);
        }
    }


    /***************************************************
     * SeparableFilter
     ***************************************************/

    void SeparableFilter::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            value_type v(r0.shape);
            cuv::alex_conv::gaussian_blur(v, p0.value.cdata(), m_kernel, true);
            cuv::alex_conv::gaussian_blur(*r0.overwrite_or_add_value(), v, m_kernel, false);
        }else if(r0.can_add_directly()){
            value_type v(r0.shape);
            cuv::alex_conv::gaussian_blur(v, p0.value.cdata(), m_kernel, true);
            cuv::alex_conv::gaussian_blur(*r0.overwrite_or_add_value(), v, m_kernel, false, 1.f, 1.f);
        }else{
            // try to overwrite p0
            value_type v(r0.shape);
            cuv::alex_conv::gaussian_blur(v, p0.value.cdata(), m_kernel, true);
            cuv::alex_conv::gaussian_blur(p0.value.data(), v, m_kernel, false);
            r0.push(p0.value);
        }
        p0.value.reset();
    }

    void SeparableFilter::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(p0.can_overwrite_directly()){
            value_type v(p0.shape);
            cuv::alex_conv::gaussian_blur(v, r0.delta.cdata(), m_kernel, true);
            cuv::alex_conv::gaussian_blur(*p0.overwrite_or_add_value(), v, m_kernel_reverse, false);
        }else if(p0.can_add_directly()){
            value_type v(p0.shape);
            cuv::alex_conv::gaussian_blur(v, r0.delta.cdata(), m_kernel_reverse, true);
            cuv::alex_conv::gaussian_blur(*p0.overwrite_or_add_value(), v, m_kernel_reverse, false, 1.f, 1.f);
        }else{
            // try to overwrite r0.delta
            value_type v(p0.shape);
            cuv::alex_conv::gaussian_blur(v, r0.delta.cdata(), m_kernel_reverse, true);
            cuv::alex_conv::gaussian_blur(r0.delta.data(), v, m_kernel_reverse, false);
            p0.push(r0.delta);
        }
        r0.delta.reset();
    }

    void SeparableFilter::_determine_shapes(){
        /*
         * images    (numFilters, imgPixY, imgPixX, numImages)
         * dst:      (numFilters, imgPixY, imgPixX, numImages)
         */
        cuvAssert(m_params[0]->shape.size()==4);
        m_results[0]->shape = m_params[0]->shape;

        unsigned int size = m_kernel.shape(0);
        m_kernel_reverse.resize(cuv::extents[size]);
        for (unsigned int i = 0; i < size; ++i)
        {
            m_kernel_reverse(i) = m_kernel(size - 1 - i);
        }
    }
}
