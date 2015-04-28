#ifndef NO_THEANO_WRAPPERS
#include "theano_ops.hpp"

namespace cuvnet
{

    void ShuffleDim::fprop(){
        using namespace cuv;
        using namespace cuv::theano_ops;
#ifdef CUVNET_USE_CPU

        throw std::runtime_error("ShuffleDim fprop: not implemented for CPU");
#else
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();
        if(r0.can_overwrite_directly()){
            dim_shuffle_vec(r0.overwrite_or_add_value().data(), inp, m_pattern);
        }else{
            value_ptr res(new value_type(inp.shape(), value_ptr::s_allocator));
            dim_shuffle_vec(*res, inp, m_pattern);

            r0.push(res); // 'copy' a newly created matrix
        }

        if(!p0.need_derivative)
            p0.value.reset();
#endif
    }


    void ShuffleDim::bprop(){
        using namespace cuv;
        using namespace cuv::theano_ops;
#ifdef CUVNET_USE_CPU
        throw std::runtime_error("ShuffleDim bprop: not implemented for CPU");
#else
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            dim_shuffle_vec(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_pattern);
        }else{
            value_ptr res(new value_type(p0.value.cdata().shape()));
            value_type& dres = *res;
            dim_shuffle_vec(dres, r0.delta.cdata(), m_pattern);
            p0.push(res);
        }
        r0.delta.reset();
        p0.value.reset(); // now we don't need it anymore ;)
#endif
    }

    void ShuffleDim::_determine_shapes(){
        assert(m_params[0]->shape.size() > 1);
        unsigned int ndim = m_pattern.size();
        std::vector<unsigned int> dst(ndim);

        for (unsigned int i = 0; i < ndim; ++i)
        {
            dst[i] = m_params[0]->shape[m_pattern[i]];
        }
        m_results[0]->shape = dst;
    }



    void FlipDims::fprop(){
        using namespace cuv;
        using namespace cuv::theano_ops;
#ifdef CUVNET_USE_CPU
        throw std::runtime_error("FlipDims fprop: not implemented for CPU");
#else
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const value_type& inp = p0.value.cdata();

        if(r0.can_overwrite_directly()){
            flip_dims_vec(r0.overwrite_or_add_value().data(), inp, m_pattern);
        }else{
            value_ptr res(new value_type(inp.shape(), value_ptr::s_allocator));
            flip_dims_vec(*res, inp, m_pattern);

            r0.push(res); // 'copy' a newly created matrix
        }

        if(!p0.need_derivative)
            p0.value.reset();
#endif
    }


    void FlipDims::bprop(){
        using namespace cuv;
        using namespace cuv::theano_ops;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

#ifdef CUVNET_USE_CPU
        throw std::runtime_error("FlipDims bprop: not implemented for CPU");
#else
        if(p0.can_overwrite_directly()){
            flip_dims_vec(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_pattern);
        }else{
            value_ptr res(new value_type(p0.value.cdata().shape()));
            value_type& dres = *res;
            flip_dims_vec(dres, r0.delta.cdata(), m_pattern);
            p0.push(res);
        }
#endif
        r0.delta.reset();
        p0.value.reset(); // now we don't need it anymore ;)
    }

    void FlipDims::_determine_shapes(){
        assert(m_params[0]->shape.size() == 4);
        m_results[0]->shape = m_params[0]->shape;
    }
    /***************************************************
     * Convolve2dTheano
     ***************************************************/

    void Convolve2dTheano::fprop(){
        using namespace cuv;
        using namespace cuv::theano_conv;

        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        cuvAssert(p0.value.cdata().is_c_contiguous());

        if(r0.can_overwrite_directly()){
            if (m_use_bias){
                value_type extended(m_extended.shape());
                param_t::element_type&  p2 = *m_params[2];
                value_type v(r0.shape);
                convolve_2d(v, p0.value.cdata(), p1.value.cdata(), m_mode);


                value_type bias = p2.value.cdata();
                bias.reshape(cuv::extents[v.shape(1) * v.shape(2) * v.shape(3)]);

                cuv::matrix_op_vec(extended,m_extended, bias, 1, BF_MULT);

                extended.reshape(v.shape());

                cuv::apply_binary_functor(*r0.overwrite_or_add_value(), v, extended, cuv::BF_ADD);
            }
            else
                convolve_2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), m_mode);
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape));
            if (m_use_bias){
                param_t::element_type&  p2 = *m_params[2];
                value_type extended(m_extended.shape());
                value_type r(r0.shape);
                convolve_2d(r, p0.value.cdata(), p1.value.cdata(), m_mode);

                value_type bias = p2.value.cdata();
                bias.reshape(cuv::extents[r.shape(1) * r.shape(2) * r.shape(3)]);

                cuv::matrix_op_vec(extended,m_extended, bias, 1, BF_MULT);
                extended.reshape(r.shape());

                cuv::apply_binary_functor(*v, r, extended, cuv::BF_ADD);
            }
            else
                convolve_2d(*v, p0.value.cdata(), p1.value.cdata(), m_mode);
            r0.push(v);
        }

        if(!p0.need_derivative && !p1.need_derivative)
        {
            p0.value.reset();
            p1.value.reset();
        }
    }


    void Convolve2dTheano::bprop(){
        using namespace cuv;
        using namespace cuv::theano_conv;

        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        assert(p0.need_derivative || p1.need_derivative);
        cuvAssert(r0.delta.cdata().is_c_contiguous());

        if(p1.need_derivative){
            // calculate p1 first, then we don't need activations
            // anymore and can overwrite them. They are usually
            // larger than the weights, so it should be better in this order.
            const value_type& delta = r0.delta.cdata();
            const value_type& img   = p0.value.cdata();
            if(p1.can_overwrite_directly()){
                d_convolve_d_kern(*p1.overwrite_or_add_value(),img, delta,  m_mode);
            }
            else{
                value_ptr ptr(new value_type(p1.shape, value_ptr::s_allocator));
                value_type& dflt = *ptr;
                d_convolve_d_kern(dflt,img, delta,  m_mode);
                p1.push(ptr);
            }
        }
        if(p0.need_derivative){
            // derivative w.r.t. images
            const value_type& delta = r0.delta.cdata();
            const value_type& flt   = p1.value.cdata();
            if(p0.can_overwrite_directly()){
                d_convolve_d_images(*p0.overwrite_or_add_value(),delta,flt, m_mode);
            }
            else{
                value_ptr ptr = p0.value;
                p0.value.reset();       // try to overwrite input activations
                value_type& v = ptr.data_onlyshape();
                d_convolve_d_images(v,delta,flt, m_mode);
                p0.push(ptr);
            }
        }
        if(m_use_bias){
            param_t::element_type&  p2 = *m_params[2];
            const value_type& delta = r0.delta.cdata();

            unsigned int size = 1;
            unsigned int ndim = delta.shape().size();
            unsigned int cols = delta.shape(0);
            for(unsigned int i = 0; i < ndim;i++){
                size *= delta.shape(i);
            }
            unsigned int rows = size / cols;

            if(p2.need_derivative){
                if(p2.can_overwrite_directly()){
                    value_type v(rows);
                    value_type r(r0.shape);
                    // multiplies delta with mask of ones at the border and zeros in center and summes up all
                    cuv::apply_binary_functor(r, delta, m_extended_orig, cuv::BF_MULT);
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_row(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[delta.shape(1)][delta.shape(2)][delta.shape(3)]);
                    *p2.overwrite_or_add_value() = v;

                }
                else if(p2.can_add_directly()){
                    value_type v(rows);
                    value_type r(r0.shape);
                    // multiplies delta with mask of ones at the border and zeros in center and summes up all
                    cuv::apply_binary_functor(r, delta, m_extended_orig, cuv::BF_MULT);
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_row(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[delta.shape(1)][delta.shape(2)][delta.shape(3)]);
                    *p2.overwrite_or_add_value() = v;

                }
                else{
                    value_ptr v(new value_type(p2.shape, value_ptr::s_allocator));
                    value_type w(rows);
                    value_type r(r0.shape);
                    // multiplies delta with mask of ones at the border and zeros in center and summes up all
                    cuv::apply_binary_functor(r, delta, m_extended_orig, cuv::BF_MULT);
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_row(w, r,RF_ADD, 1.f, 0.f);
                    w.reshape(cuv::extents[delta.shape(1)][delta.shape(2)][delta.shape(3)]);
                    *v = w;
                    p2.push(v);
                }

            }
            p2.value.reset();
        }


        p0.value.reset();
        p1.value.reset();
        r0.delta.reset();
    }

    void Convolve2dTheano::_determine_shapes(){
        //dst       (nImg, nFilt, nModules, nModules)
        //img       (nImg, nImgChan, nImgPiY, nImgPix)
        //filter    (nFilt,nFiltChan, nFiltPiY,nFiltPix)


        assert(m_params[0]->shape.size()==4);
        assert(m_params[1]->shape.size()==4);
        std::vector<unsigned int> dst(4);
        const std::vector<unsigned int>& img = m_params[0]->shape;
        const std::vector<unsigned int>& flt = m_params[1]->shape;
        unsigned int nFilt    = flt[0];
        unsigned int nImgPixY = img[2];
        unsigned int nImgPixX = img[3];
        unsigned int nFltPixY = flt[2];
        unsigned int nFltPixX = flt[3];

        unsigned int nOutPixX = m_mode == "valid" ? nImgPixX+1-nFltPixX :  nImgPixX - 1 + nFltPixX;
        unsigned int nOutPixY = m_mode == "valid" ? nImgPixY+1-nFltPixY :  nImgPixY - 1 + nFltPixY;

        if(m_use_bias){
            // create mask, which is used for delta calcualtion of bias
            unsigned int width_x = nFltPixX-1;
            unsigned int width_y = nFltPixY-1;
            unsigned int mask_size_x = nImgPixX + nFltPixX - 1;
            unsigned int mask_size_y = nImgPixY + nFltPixY - 1;
            value_type mask(cuv::extents[mask_size_y][mask_size_x]);
            for (unsigned int i = 0; i < mask_size_y; ++i)
            {
                for (unsigned int j = 0; j < mask_size_x; ++j){
                    if (j < width_x || j >= nImgPixX || i < width_y || i >= nImgPixY)
                        mask(i,j) = 1;
                    else
                        mask(i,j) = 0;
                }
            }
            mask.reshape(cuv::extents[mask_size_y * mask_size_x]);
            m_extended_orig.resize(cuv::extents[img[0]][nFilt][mask_size_y * mask_size_x]);
            m_extended_orig = 1.f;
            cuv::matrix_op_vec(m_extended_orig, m_extended_orig, mask, 2, cuv::BF_MULT);
            m_extended = m_extended_orig.copy();
            m_extended.reshape(cuv::extents[img[0]][nFilt * mask_size_y * mask_size_x]);

            m_extended_orig.reshape(cuv::extents[img[0]][nFilt][mask_size_y][mask_size_x]);
        }

        dst[0] = img[0];
        dst[1] = nFilt;
        dst[2] = nOutPixY;
        dst[3] = nOutPixX;
        m_results[0]->shape = dst;
    }
}
#endif
