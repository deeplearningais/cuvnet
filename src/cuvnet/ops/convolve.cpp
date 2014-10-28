#include <cuvnet/common.hpp>
#include <cuvnet/op_utils.hpp>
#include "convolve.hpp"

namespace cuvnet
{
    void Convolve::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.color = "chartreuse4";
        if(m_params[0]->shape.size() > 0 && m_params[1]->shape.size() > 0){
            desc.label = "Conv (" +
                boost::lexical_cast<std::string>(m_params[0]->shape[0]) + "/" +
                boost::lexical_cast<std::string>(m_nGroups) + ":" +
                boost::lexical_cast<std::string>(m_results[0]->shape[0]) + " fs" +
                boost::lexical_cast<std::string>((int)sqrt(m_params[1]->shape[1])) + ")";
        }else{
            desc.label = "Conv";
        }
    }
    void Convolve::set_random_sparse(unsigned int nFiltChan){
        determine_shapes(*this);

        int nImgChan  = m_params[0]->shape[0];
        if(nFiltChan == 0)
            nFiltChan = nImgChan / m_nGroups;
        cuvAssert(nFiltChan * m_nGroups % nImgChan == 0);
        int nGroups   = m_nGroups;
        int oversample = nGroups * nFiltChan / nImgChan;

        unsigned int nFiltChanP0 = m_params[1]->shape[0];
        cuvAssert(nFiltChanP0 == nFiltChan);
        //int nFltPix  = m_params[1]->shape[1];
        //int nDstMaps = m_params[1]->shape[2];
        //m_weights->data().resize(cuv::extents[nFiltChan][nFltPix][nDstMaps]);

        m_indices.resize(cuv::extents[nGroups][oversample * nImgChan]);
        for(int i=0; i < nGroups; i++){
            std::vector<int> v(nImgChan);
            for (unsigned int k = 0; k < v.size(); ++k)
                v[k] = k;
            // we shouldn't shuffle in this test to get same result as dense connection
            std::random_shuffle(v.begin(), v.end());
            for (int o = 0; o < oversample; ++o)
            {
                for (int k = 0; k < nImgChan; ++k)
                {
                    m_indices(i, o*nImgChan + k) = v[k];
                }
            }
        }
    }
    void Convolve::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        cuvAssert(p0.value.cdata().is_c_contiguous());

        bool filtSizeOK = (r0.shape[0] % 16) == 0;
        bool rnd = m_indices.ptr() != NULL;

        if(filtSizeOK && r0.can_overwrite_directly()){
            if(!rnd)
                convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), m_padding_start,m_stride,m_nGroups);
            else
                convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), m_indices, m_padding_start,m_stride,m_nGroups);
        }else if(filtSizeOK && r0.can_add_directly()){
            if(!rnd)
                convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), m_padding_start,m_stride,m_nGroups, 1.f,1.f);
            else
                convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), m_indices, m_padding_start,m_stride,m_nGroups, 1.f,1.f);
        }else{
            // reallocate *sigh*
            if(filtSizeOK){
                value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
                if(!rnd)
                    convolve2d(*v, p0.value.cdata(), p1.value.cdata(), m_padding_start, m_stride,m_nGroups);
                else
                    convolve2d(*v, p0.value.cdata(), p1.value.cdata(), m_indices, m_padding_start, m_stride,m_nGroups);
                r0.push(v);

            }else{
                // Alex' code has some serious restrictions; the one hurting me most
                // is about the number of output maps (n%16==0).
                // I'll emulate a less restricted version at some expense here
                // by creating larger arrays if necessary>
                unsigned int nFiltReal = r0.shape[0];
                unsigned int nFiltTmp  = 16 * std::ceil(nFiltReal / 16.);                            // create intermediate representation of the outputs
                value_type tmp_dst(extents[nFiltTmp][r0.shape[1]][r0.shape[2]][r0.shape[3]]);

                // create intermediate copy of weights
                value_type tmp_flt(extents[p1.shape[0]][p1.shape[1]][nFiltTmp]);
                tmp_flt = 0.f;
                //tmp_flt[indices[index_range()][index_range()][index_range(0,nFiltTmp)]] = p1.value.cdata().copy();
                tensor_view<float, cuv::dev_memory_space> wview(tmp_flt,
                        indices[index_range()][index_range()][index_range(0,nFiltReal)]);
                wview = p1.value.cdata();

                if(!rnd)
                    convolve2d(tmp_dst, p0.value.cdata(), tmp_flt, m_padding_start, m_stride,m_nGroups);
                else
                    convolve2d(tmp_dst, p0.value.cdata(), tmp_flt, m_indices, m_padding_start, m_stride,m_nGroups);
                value_ptr vp(new value_type(tmp_dst[indices[index_range(0,nFiltReal)][index_range()][index_range()][index_range()]]));
                r0.push(vp);
            }
        }
        if(!p0.need_derivative && !p1.need_derivative)
        {
            p0.value.reset();
            p1.value.reset();
        }else{
            // if e.g. p0.need_derivative, then we would not need
            // p0.value, but we might just as well overwrite it
            // in backprop stage. If space is an issue, we can
            // also delete it.
        }
    }

    void Convolve::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;

        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        assert(p0.need_derivative || p1.need_derivative);
        cuvAssert(r0.delta.cdata().is_c_contiguous());

        // Alex' code has some serious restrictions; the one hurting me most
        // is about the number of output maps (n%16==0).
        // I'll emulate a less restricted version at some expense here
        // by creating larger arrays if necessary>
        bool filtSizeOK = (r0.shape[0] % 16) == 0;
        unsigned int nFiltReal = r0.shape[0];
        unsigned int nFiltTmp  = 16 * ceil(nFiltReal / 16.);                            // create intermediate representation of the outputs
        boost::scoped_ptr<value_type> tmp_r0delta;
        boost::scoped_ptr<value_type> tmp_w;
        boost::scoped_ptr<value_type> tmp_dw;

        bool rnd = m_indices.ptr() != NULL;

        if(!filtSizeOK){
            // create intermediate copy of deltas
            tmp_r0delta.reset(new value_type(extents[nFiltTmp][r0.shape[1]][r0.shape[2]][r0.shape[3]], value_ptr::s_allocator));
            {
                *tmp_r0delta = 0.f;
                (*tmp_r0delta)[indices[index_range(0,nFiltReal)][index_range()][index_range()][index_range()]] = r0.delta.cdata();
            }

            // create intermediate copy of weights
            tmp_w.reset(new value_type(extents[p1.shape[0]][p1.shape[1]][nFiltTmp], value_ptr::s_allocator));
            {
                *tmp_w = 0.f;
                (*tmp_w)[indices[index_range()][index_range()][index_range(0,nFiltReal)]] = p1.value.cdata().copy();
            }

            // create intermediate representation of filter derivative
            tmp_dw.reset(new value_type(extents[p1.shape[0]][p1.shape[1]][nFiltTmp], value_ptr::s_allocator));
        }

        if(p1.need_derivative){
            // calculate p1 first, then we don't need activations
            // anymore and can overwrite them. They are usually
            // larger than the weights, so it should be better in this order.
            const value_type& delta = r0.delta.cdata();
            const value_type& img   = p0.value.cdata();
           if(filtSizeOK){
               if(p1.can_overwrite_directly()){
                   if(!rnd)
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, m_padding_start, m_stride, m_nGroups,m_partial_sum);
                   else
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, m_indices, m_padding_start, m_stride, m_nGroups,m_partial_sum);
               }
               else if(p1.can_add_directly()){
                   if(!rnd)
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, m_padding_start, m_stride, m_nGroups,m_partial_sum, 1.f,1.f);
                   else
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, m_indices, m_padding_start, m_stride, m_nGroups,m_partial_sum, 1.f,1.f);
               }else{
                   value_ptr ptr(new value_type(p1.shape, value_ptr::s_allocator));
                   value_type& dflt = *ptr;
                   if(!rnd)
                       d_conv2d_dfilt(dflt,delta,img, m_padding_start, m_stride, m_nGroups,m_partial_sum);
                   else
                       d_conv2d_dfilt(dflt,delta,img, m_indices, m_padding_start, m_stride, m_nGroups,m_partial_sum);
                   p1.push(ptr);
               }


           }else{
               /* THIS DOES NOT WORK!
               if(p1.can_overwrite_directly()){
                   if(!rnd)
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, m_padding_start, m_stride, m_nGroups,m_partial_sum);
                   else
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, m_indices, m_padding_start, m_stride, m_nGroups,m_partial_sum);
               }
               else if(p1.can_add_directly()){
                   if(!rnd)
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, m_padding_start, m_stride, m_nGroups,m_partial_sum, 1.f,1.f);
                   else
                       d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, m_indices, m_padding_start, m_stride, m_nGroups,m_partial_sum, 1.f,1.f);
               }else{*/
                   value_type& dflt = *tmp_dw;
                   if(!rnd)
                       d_conv2d_dfilt(dflt,*tmp_r0delta,img, m_padding_start, m_stride, m_nGroups,m_partial_sum);
                   else
                       d_conv2d_dfilt(dflt,*tmp_r0delta,img, m_indices, m_padding_start, m_stride, m_nGroups,m_partial_sum);
                   value_ptr ptr(new value_type(dflt[indices[index_range()][index_range()][index_range(0,nFiltReal)]].copy()));
                   p1.push(ptr);
               //}

           }
        }

        if(p0.need_derivative){
            // derivative w.r.t. images
            const value_type& delta = r0.delta.cdata();
            const value_type& flt   = p1.value.cdata();
           if(filtSizeOK){
               if(p0.can_overwrite_directly()){
                   if(!rnd)
                       d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, m_padding_start, m_stride, m_nGroups);
                   else
                       d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, m_indices, m_padding_start, m_stride, m_nGroups);
               }
               else if(p0.can_add_directly()){
                   if(!rnd)
                       d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, m_padding_start, m_stride, m_nGroups,  1.f,1.f);
                   else
                       d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, m_indices, m_padding_start, m_stride, m_nGroups,  1.f,1.f);
               }else{
                   value_ptr ptr = p0.value;
                   p0.value.reset();       // try to overwrite input activations
                   value_type& v = ptr.data_onlyshape();
                   if(!rnd)
                       d_conv2d_dimg(v, delta, flt, m_padding_start,m_stride,m_nGroups);
                   else
                       d_conv2d_dimg(v, delta, flt, m_indices, m_padding_start,m_stride,m_nGroups);
                   p0.push(ptr);
               }


           }else{
               if(p0.can_overwrite_directly()){
                   if(!rnd)
                       d_conv2d_dimg(*p0.overwrite_or_add_value(), *tmp_r0delta, *tmp_w, m_padding_start,m_stride,m_nGroups);
                   else
                       d_conv2d_dimg(*p0.overwrite_or_add_value(), *tmp_r0delta, *tmp_w, m_indices, m_padding_start,m_stride,m_nGroups);
               }
               else if(p0.can_add_directly()){
                   if(!rnd)
                       d_conv2d_dimg(*p0.overwrite_or_add_value(), *tmp_r0delta, *tmp_w, m_padding_start,m_stride,m_nGroups, 1.f, 1.f);
                   else
                       d_conv2d_dimg(*p0.overwrite_or_add_value(), *tmp_r0delta, *tmp_w, m_indices, m_padding_start,m_stride,m_nGroups, 1.f, 1.f);
               }else{
                   value_ptr ptr = p0.value;
                   p0.value.reset();       // try to overwrite input activations
                   value_type& v = ptr.data_onlyshape();
                   if(!rnd)
                       d_conv2d_dimg(v, *tmp_r0delta, *tmp_w, m_padding_start,m_stride,m_nGroups);
                   else
                       d_conv2d_dimg(v, *tmp_r0delta, *tmp_w, m_indices, m_padding_start,m_stride,m_nGroups);
                   p0.push(ptr);
               }
           }
        }

        p0.value.reset();
        p1.value.reset();
        r0.delta.reset();
    }

    void Convolve::_determine_shapes(){
        /*
         *  dst       (nFilt, nModules, nImg)
         *  img       (nImgChan, nImgPix, nImg)
         *  filter    (nFiltChan, nFiltPix, nFilt)
         */


        assert(m_params[0]->shape.size()==4);
        assert(m_params[1]->shape.size()==3);
        std::vector<unsigned int> dst(4);
        const std::vector<unsigned int>& img = m_params[0]->shape;
        const std::vector<unsigned int>& flt = m_params[1]->shape;
        unsigned int nFilt    = flt[2];
        unsigned int nImgPixY = img[1];
        unsigned int nImgPixX = img[2];
        unsigned int nFltPixX = sqrt(flt[1]);
        assert(nFltPixX*nFltPixX==flt[1]);

        if(m_padding_start) // set to `1' in constructor when padding requested
            {
        		//m_padding_start = -(int)nFltPixX/2; // assume nFltPixX%2==1
        		m_padding_start = -m_padding_size;
            }

        // force symmetric padding
        int padsize = m_padding_size;
        if(m_symmetric_padding)
            padsize *= 2;

#define DIVUP(x,y) (((x)+ (y) -1) / (y))
        unsigned int nOutPixX = is_padded()
            ? DIVUP(nImgPixX+padsize-nFltPixX, m_stride)+1
            : DIVUP(nImgPixX        -nFltPixX, m_stride)+1;
        unsigned int nOutPixY = is_padded()
            ? DIVUP(nImgPixY+padsize-nFltPixX, m_stride)+1
            : DIVUP(nImgPixY        -nFltPixX, m_stride)+1;

        log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("determine_shapes"));
        LOG4CXX_WARN(log, "Convolving image of shape ("
                << boost::lexical_cast<std::string>(nFilt)
                << " x " << boost::lexical_cast<std::string>(nImgPixY)
                << " x " << boost::lexical_cast<std::string>(nImgPixX)
                << " x " << boost::lexical_cast<std::string>(img[3])
                << ") to shape ("
                << boost::lexical_cast<std::string>(nFilt)
                << " x " << boost::lexical_cast<std::string>(nOutPixY)
                << " x " << boost::lexical_cast<std::string>(nOutPixX)
                << " x " << boost::lexical_cast<std::string>(img[3])
                << ") using filters of size " << boost::lexical_cast<std::string>(nFltPixX)
                << "padsize: " << padsize << " padstart: " << m_padding_start);

        dst[0] = nFilt;
        dst[1] = nOutPixY;
        dst[2] = nOutPixX;
        dst[3] = img[3];
        if(m_partial_sum){
            cuvAssert((nOutPixX * nOutPixY) % m_partial_sum == 0);
        }
        m_results[0]->shape = dst;
    }




#ifndef NO_THEANO_WRAPPERS
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


#endif /* NO_THEANO_WRAPPERS */


    /***************************************************
     * BedOfNails
     ***************************************************/

    void BedOfNails::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            bed_of_nails(*r0.overwrite_or_add_value(), p0.value.cdata(), m_startx,m_stridex);
        }else if(r0.can_add_directly()){
            bed_of_nails(*r0.overwrite_or_add_value(), p0.value.cdata(), m_startx,m_stridex, 1.f,1.f);
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            bed_of_nails(*v, p0.value.cdata(), m_startx,m_stridex);
            r0.push(v);
        }
        p0.value.reset();
    }

    void BedOfNails::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);
        if(p0.can_overwrite_directly()){
            bed_of_nails_grad(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_startx,m_stridex);
        }else if(p0.can_add_directly()){
            bed_of_nails_grad(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_startx,m_stridex, 1.f, 1.f);
        }else{
            value_ptr ptr(new value_type(p0.shape, value_ptr::s_allocator));
            bed_of_nails_grad(*ptr, r0.delta.cdata(), m_startx,m_stridex);
            p0.push(ptr);
        }
        r0.delta.reset();
    }

    void BedOfNails::_determine_shapes(){
        /*
         * images    (numFilters, imgPixY, imgPixX, numImages)
         * dst:      (numFilters, outputs, numImages)
         */
        assert(m_params[0]->shape.size()==4);
        std::vector<unsigned int> img = m_params[0]->shape;
        cuvAssert(img[1]==img[2]); // currently, cudaConv2 only supports square images for subsampling

        std::vector<unsigned int> dst(4);
        dst[0] = img[0];
        dst[1] = (img[1]-m_startx) / m_stridex;
        dst[2] = (img[2]-m_startx) / m_stridex;
        dst[3] = img[3];
        m_results[0]->shape = dst;
    }

    /***************************************************
     * ResizeBilinear
     ***************************************************/
    void ResizeBilinear::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            resize_bilinear(*r0.overwrite_or_add_value(), p0.value.cdata(), m_scale);
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            resize_bilinear(*v, p0.value.cdata(), m_scale);
            r0.push(v);
        }
        p0.value.reset();
    }

    void ResizeBilinear::bprop(){
        std::runtime_error("Bprop of ResizeBilinear does not work: Upscaling is not correctly implemented in CudaConv2");
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);
        if(p0.can_overwrite_directly()){
            resize_bilinear(*p0.overwrite_or_add_value(), r0.delta.cdata(), 1.f/m_scale);
        }else{
            value_ptr ptr(new value_type(p0.shape, value_ptr::s_allocator));
            resize_bilinear(*ptr, r0.delta.cdata(), 1.f/m_scale);
            p0.push(ptr);
        }
        r0.delta.reset();
    }

    void ResizeBilinear::_determine_shapes(){
        /*
         * images    (numFilters, imgPixY, imgPixX, numImages)
         * dst:      (numFilters, outputs, numImages)
         */
        assert(m_params[0]->shape.size()==4);
        std::vector<unsigned int> img = m_params[0]->shape;
        cuvAssert(img[1]==img[2]); // currently, cudaConv2 only supports square images for subsampling

        std::vector<unsigned int> dst(4);
        dst[0] = img[0];
        dst[1] = (unsigned int)(img[1] / m_scale + 0.5);
        dst[2] = (unsigned int)(img[2] / m_scale + 0.5);
        dst[3] = img[3];
        m_results[0]->shape = dst;
    }

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


    /***************************************************
     * ResponseNormalizationCrossMaps
     ***************************************************/

    void ResponseNormalizationCrossMaps::release_data(){
        m_denom.dealloc();
        m_orig_out.reset();
        Op::release_data();
    }

    void ResponseNormalizationCrossMaps::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        m_denom.resize(r0.shape);
        if(r0.can_overwrite_directly()){
            // note: we need to /first/ run the function, /then/ copy the cow_ptr!
            //       otherwise only a copy will be overwritten.
            cuv::alex_conv::response_norm_cross_map(*r0.overwrite_or_add_value(), m_denom, p0.value.cdata(), m_group_size, m_add_scale, m_pow_scale, m_blocked);
            m_orig_out = r0.overwrite_or_add_value();
        }else{
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            cuv::alex_conv::response_norm_cross_map(*v, m_denom, p0.value.cdata(), m_group_size, m_add_scale, m_pow_scale, m_blocked);
            r0.push(v);
            m_orig_out = v;
        }
        if(!p0.need_derivative) {
            p0.value.reset();
            m_denom.dealloc();
            m_orig_out.reset();
        }
    }

    void ResponseNormalizationCrossMaps::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(p0.can_overwrite_directly()){
            cuv::alex_conv::response_norm_cross_map_grad(*r0.overwrite_or_add_value(), *m_orig_out, p0.value.cdata(), r0.delta.cdata(), m_denom, m_group_size, m_add_scale, m_pow_scale, m_blocked);
        }else if(p0.can_add_directly()){
            cuv::alex_conv::response_norm_cross_map_grad(*r0.overwrite_or_add_value(), *m_orig_out, p0.value.cdata(), r0.delta.cdata(), m_denom, m_group_size, m_add_scale, m_pow_scale, m_blocked, 1.f, 1.f);
        }else{
            // try to overwrite r0.delta
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            cuv::alex_conv::response_norm_cross_map_grad(*v, *m_orig_out, p0.value.cdata(), r0.delta.cdata(), m_denom, m_group_size, m_blocked, m_add_scale, m_pow_scale);
            p0.push(v);
        }
        r0.delta.reset();
        m_orig_out.reset();
        m_denom.dealloc();
    }


    void ResponseNormalizationCrossMaps::_determine_shapes(){
        /*
         * images    (numFilters, imgPixY, imgPixX, numImages)
         * dst:      (numFilters, imgPixY, imgPixX, numImages)
         */
        assert(m_params[0]->shape.size()==4);
        if(m_group_size <= 0)
            m_group_size = m_params[0]->shape[0];
        m_results[0]->shape = m_params[0]->shape;
    }


    /***************************************************
     * ResponseNormalization
     ***************************************************/

    void ResponseNormalization::release_data(){
        m_denom.dealloc();
        m_orig_out.reset();
        Op::release_data();
    }

    void ResponseNormalization::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        m_denom.resize(r0.shape);
        if(r0.can_overwrite_directly()){
            // note: we need to /first/ run the function, /then/ copy the cow_ptr!
            //       otherwise only a copy will be overwritten.
            cuv::alex_conv::response_normalization(*r0.overwrite_or_add_value(), m_denom, p0.value.cdata(), m_patch_size, m_add_scale, m_pow_scale);
            m_orig_out = r0.overwrite_or_add_value();
        }else{
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            cuv::alex_conv::response_normalization(*v, m_denom, p0.value.cdata(), m_patch_size, m_add_scale, m_pow_scale);
            r0.push(v);
            m_orig_out = v;
        }
        if(!p0.need_derivative) {
            p0.value.reset();
            m_denom.dealloc();
            m_orig_out.reset();
        }
    }

    void ResponseNormalization::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(p0.can_overwrite_directly()){
            response_normalization_grad(*r0.overwrite_or_add_value(), *m_orig_out, p0.value.cdata(), r0.delta.cdata(), m_denom, m_patch_size, m_add_scale, m_pow_scale);
        }else if(p0.can_add_directly()){
            response_normalization_grad(*r0.overwrite_or_add_value(), *m_orig_out, p0.value.cdata(), r0.delta.cdata(), m_denom, m_patch_size, m_add_scale, m_pow_scale, 1.f, 1.f);
        }else{
            // try to overwrite r0.delta
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            response_normalization_grad(*v, *m_orig_out, p0.value.cdata(), r0.delta.cdata(), m_denom, m_patch_size, m_add_scale, m_pow_scale);
            p0.push(v);
        }
        r0.delta.reset();
        m_orig_out.reset();
        m_denom.dealloc();
    }


    void ResponseNormalization::_determine_shapes(){
        /*
         * images    (numFilters, imgPixY, imgPixX, numImages)
         * dst:      (numFilters, imgPixY, imgPixX, numImages)
         */
        assert(m_params[0]->shape.size()==4);
        m_results[0]->shape = m_params[0]->shape;
    }

    /***************************************************
     * ContrastNormalization
     ***************************************************/
    void ContrastNormalization::release_data(){
        m_denom.dealloc();
        m_meandiffs.dealloc();
        m_orig_out.reset();
        Op::release_data();
    }
    void ContrastNormalization::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        m_denom.resize(r0.shape);
        m_meandiffs.resize(r0.shape);
        if(r0.can_overwrite_directly()){
            // note: we need to /first/ run the function, /then/ copy the cow_ptr!
            //       otherwise only a copy will be overwritten.
            cuv::alex_conv::contrast_normalization(*r0.overwrite_or_add_value(), m_denom, m_meandiffs, p0.value.cdata(), m_patch_size, m_add_scale, m_pow_scale);
            m_orig_out = r0.overwrite_or_add_value();
        }else{
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            cuv::alex_conv::contrast_normalization(*v, m_denom, m_meandiffs, p0.value.cdata(), m_patch_size, m_add_scale, m_pow_scale);
            r0.push(v);
            m_orig_out = v;
        }
        p0.value.reset();
        if(!p0.need_derivative) {
            m_denom.dealloc();
            m_orig_out.reset();
            m_meandiffs.dealloc();
        }
    }
    void ContrastNormalization::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(p0.can_overwrite_directly()){
            contrast_normalization_grad(*r0.overwrite_or_add_value(), *m_orig_out, m_meandiffs, r0.delta.cdata(), m_denom, m_patch_size, m_add_scale, m_pow_scale);
        }else if(p0.can_add_directly()){
            contrast_normalization_grad(*r0.overwrite_or_add_value(), *m_orig_out, m_meandiffs, r0.delta.cdata(), m_denom, m_patch_size, m_add_scale, m_pow_scale, 1.f, 1.f);
        }else{
            // try to overwrite r0.delta
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            contrast_normalization_grad(*v, *m_orig_out, m_meandiffs, r0.delta.cdata(), m_denom, m_patch_size, m_add_scale, m_pow_scale);
            p0.push(v);
        }
        r0.delta.reset();
        m_orig_out.reset();
        m_meandiffs.dealloc();
        m_denom.dealloc();
    }
    void ContrastNormalization::_determine_shapes(){
        /*
         * images    (numFilters, imgPixY, imgPixX, numImages)
         * dst:      (numFilters, imgPixY, imgPixX, numImages)
         */
        assert(m_params[0]->shape.size()==4);
        m_results[0]->shape = m_params[0]->shape;
    }

    /***************************************************
     * LocalPooling
     ***************************************************/
    void LocalPooling::release_data(){
        m_result.reset();
        Op::release_data();
    }

    void LocalPooling::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        //unsigned int outy = p0.shape[1]/m_subsx;
        unsigned int outx = p0.shape[2]/m_subsx;
        if(r0.can_overwrite_directly()){
            //int subsX, int startX, int strideX, int outputsX, pool_type pooler
            local_pool(*r0.overwrite_or_add_value(),p0.value.cdata(),
                    m_subsx, m_startx, m_stridex, outx, m_pooltype);
            if(m_pooltype == PT_MAX && p0.need_derivative)
                m_result = r0.overwrite_or_add_value(); // save for bprop
        }else{
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            local_pool(*v,p0.value.cdata(),
                    m_subsx, m_startx, m_stridex, outx, m_pooltype);
            r0.push(v);
            if(m_pooltype == PT_MAX && p0.need_derivative)
                m_result = v; // save for bprop
        }
        if(m_pooltype == PT_AVG || !p0.need_derivative){
            p0.value.reset();
            // if memory is not an issue, we can also leave it here
            // and write to it in the backward pass.
        }
        else{
            // keep p0, needed for bprop of PT_MAX
        }
    }

    void LocalPooling::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        //void local_max_pool_grad(target, const images, const maxGrads,
        //        const maxActs, int subsX, int startX, int strideX, float factNew=1.f, float factOld=0.f);

        //void local_avg_pool_grad(target, const avgGrads,
        //        int subsX, int startX, int strideX);

        // bprop with alex' code only works if filtSizeOK:
        bool filtSizeOK = (r0.shape[0] % 16) == 0;

        if(filtSizeOK){
            if(m_pooltype == PT_AVG){
                if(p0.can_overwrite_directly()){
                    local_avg_pool_grad(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_subsx,m_startx,m_stridex);
                }else{
                    // try overwriting p0.
                    //value_ptr ptr = p0.value;
                    value_ptr ptr(new value_type(p0.shape, value_ptr::s_allocator));
                    value_type& v = *ptr;
                    local_avg_pool_grad(v, r0.delta.cdata(), m_subsx,m_startx,m_stridex);
                    p0.push(ptr);
                }
            }else if(m_pooltype == PT_MAX){
                if(p0.can_overwrite_directly()){
                    local_max_pool_grad(*p0.overwrite_or_add_value(), p0.value.cdata(), r0.delta.cdata(), m_result.cdata(), m_subsx,m_startx,m_stridex);
                }else if(p0.can_add_directly()){
                    local_max_pool_grad(*p0.overwrite_or_add_value(), p0.value.cdata(), r0.delta.cdata(), m_result.cdata(), m_subsx,m_startx,m_stridex, 1.f,1.f);
                }else{
                    value_ptr ptr(new value_type(p0.shape, value_ptr::s_allocator));
                    value_type& v = *ptr;
                    local_max_pool_grad(v, p0.value.cdata(), r0.delta.cdata(), m_result.cdata(), m_subsx,m_startx,m_stridex);
                    p0.push(ptr);
                }
                p0.value.reset();
                m_result.reset();
            }
        }else{
            // number of maps was not a multiple of 16 --> we need to pad the data.
            unsigned int nFiltReal = p0.shape[0];
            unsigned int nFiltTmp  = 16 * std::ceil(nFiltReal / 16.);                            // create intermediate representation of the outputs
            value_type tmp_p0delta(extents[nFiltTmp][p0.shape[1]][p0.shape[2]][p0.shape[3]]);
            value_type tmp_r0delta(extents[nFiltTmp][r0.shape[1]][r0.shape[2]][r0.shape[3]]);
            typedef index_range range;
            tmp_r0delta[
                    indices[range(0, nFiltReal)][range()][range()][range()]] = r0.delta.cdata();

            if(m_pooltype == PT_AVG){
                local_avg_pool_grad(tmp_p0delta, tmp_r0delta, m_subsx,m_startx,m_stridex);
            }else if(m_pooltype == PT_MAX){
                // pad input
                value_type tmp_p0value(extents[nFiltTmp][p0.shape[1]][p0.shape[2]][p0.shape[3]]);
                tmp_p0value[
                    indices[range(0,nFiltReal)][range()][range()][range()]] = p0.value.cdata();
                // pad result
                value_type tmp_r0value(extents[nFiltTmp][r0.shape[1]][r0.shape[2]][r0.shape[3]]);
                tmp_r0value[
                        indices[range(0,nFiltReal)][range()][range()][range()]] = m_result.cdata();
                local_max_pool_grad(tmp_p0delta, tmp_p0value, tmp_r0delta, tmp_r0value, m_subsx,m_startx,m_stridex);
            }
            p0.push(value_ptr(
                        new value_type(
                            tmp_p0delta[indices[range(0,nFiltReal)][range()][range()][range()]])));
        }
        p0.value.reset();
        r0.delta.reset();
        m_result.reset();
    }

    void LocalPooling::_graphviz_node_desc(detail::graphviz_node& desc)const{
        using namespace cuv::alex_conv;
        if(m_pooltype == PT_MAX)
            desc.label = "Max";
        else if(m_pooltype == PT_AVG)
            desc.label = "Avg";
        desc.label += "Pool (size" + boost::lexical_cast<std::string>(m_subsx) + ", stride"
            +                    boost::lexical_cast<std::string>(m_stridex) + ", start"
            +                    boost::lexical_cast<std::string>(m_startx) + ")";
    }

    void LocalPooling::_determine_shapes(){
        /*
         * images    (numFilters, imgPixY, imgPixX, numImages)
         * dst:      (numFilters, outputs, numImages)
         */
        assert(m_params[0]->shape.size()==4);
        std::vector<unsigned int> img = m_params[0]->shape;
        std::vector<unsigned int> dst(4);
        dst[0] = img[0];
        if ( img[1] <= m_subsx ){
            dst[1] = 1;
            m_subsx -= m_subsx - img[1];
        }
        else
            dst[1] = (img[1] - m_startx) / m_stridex;

        if ( img[2] <= m_subsx ){
            dst[2] = 1;
            m_subsx -= m_subsx - img[2];
        }
        else
            dst[2] = (img[2] - m_startx) / m_stridex;
        dst[3] = img[3];

        log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("determine_shapes"));
        LOG4CXX_WARN(log, "Pooling image of shape ("
                +         boost::lexical_cast<std::string>(img[0])
                + " x " + boost::lexical_cast<std::string>(img[1])
                + " x " + boost::lexical_cast<std::string>(img[2])
                + " x " + boost::lexical_cast<std::string>(img[3])
                + ") to shape ("
                +         boost::lexical_cast<std::string>(dst[0])
                + " x " + boost::lexical_cast<std::string>(dst[1])
                + " x " + boost::lexical_cast<std::string>(dst[2])
                + " x " + boost::lexical_cast<std::string>(dst[3])
                + ")");

        cuvAssert(img[1]==img[2]); // currently, cudaConv2 only supports square images for pooling
//        if(m_subsx * dst[1] != img[1]){
//            throw std::runtime_error(
//                    "LocalPooling: incoming size `"
//                    + boost::lexical_cast<std::string>(img[1])
//                    + "' is not divisible by subsampling factor `"
//                    + boost::lexical_cast<std::string>(m_subsx));
//        }
//        cuvAssert(m_subsx * dst[2] == img[2]);
        m_results[0]->shape = dst;
    }

    void ReorderForConv::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            reorder_for_conv(*r0.overwrite_or_add_value(), p0.value.cdata());
        }else{
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            reorder_for_conv(*v, p0.value.cdata());
            r0.push(v);
        }
        p0.value.reset();
    }

    void ReorderForConv::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            reorder_from_conv(*p0.overwrite_or_add_value(),r0.delta.cdata());
        }else{
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            reorder_from_conv(*v,r0.delta.cdata());
            p0.push(v);
        }
        r0.delta.reset();
    }

    void ReorderForConv::_determine_shapes(){
        assert(m_params[0]->shape.size()==4);
        const std::vector<unsigned int>& img = m_params[0]->shape;
        std::vector<unsigned int> dst(4);
        dst[0] = img[1];
        dst[1] = img[2];
        dst[2] = img[3];
        dst[3] = img[0];
        m_results[0]->shape = dst;
    }

    /***************************************************
     * ReorderFromConv
     ***************************************************/

    void ReorderFromConv::fprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            reorder_from_conv(*r0.overwrite_or_add_value(), p0.value.cdata());
        }else{
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
            reorder_from_conv(*v, p0.value.cdata());
            r0.push(v);
        }
        p0.value.reset();
    }

    void ReorderFromConv::bprop(){
        using namespace cuv;
        using namespace cuv::alex_conv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            reorder_for_conv(*p0.overwrite_or_add_value(),r0.delta.cdata());
        }else{
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            reorder_for_conv(*v,r0.delta.cdata());
            p0.push(v);
        }
        r0.delta.reset();
    }

    void ReorderFromConv::_determine_shapes(){
        assert(m_params[0]->shape.size()==4);
        const std::vector<unsigned int>& img = m_params[0]->shape;
        std::vector<unsigned int> dst(4);
        dst[0] = img[3];
        dst[1] = img[0];
        dst[2] = img[1];
        dst[3] = img[2];
        m_results[0]->shape = dst;
    }

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
