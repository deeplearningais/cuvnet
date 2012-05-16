#ifndef __OP_CONVOLVE_HPP__
#     define __OP_CONVOLVE_HPP__

#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
namespace cuvnet
{

    /**
     * convolve a set of images using a set of filters 
     * This is the "neural net" type convolution, where filters are 3D and
     * results are summed over input channels.
     *
     * First param, the images, must have shape 
     *
     *  nChannels x nPixels x nImages
     *
     * while filters have shape
     *  
     *  nFiltChannels x nFiltPix x nFilt
     *
     */
    class Convolve
        :public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_nGroups;
                unsigned int m_partial_sum;
            public:
                Convolve() :Op(2,1){} // for serialization

                /**
                 * constructor.
                 *
                 * @param images nChannels x nPixels x nImages
                 * @param filters nFiltChannels x nFiltPix x nFilt
                 */
                Convolve(result_t& images, result_t& filters)
                    :Op(2,1)
                    ,m_nGroups(1)
                    ,m_partial_sum(1)
                {
                    add_param(0,images);
                    add_param(1,filters);
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    cuvAssert(p0.value.cdata().is_c_contiguous());

                    bool filtSizeOK = (r0.shape[0] % 16) == 0;

                    if(filtSizeOK && r0.can_overwrite_directly()){
                        convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), 0,1,m_nGroups);
                    }else if(filtSizeOK && r0.can_add_directly()){
                        convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), 0,1,m_nGroups, 1.f,1.f);
                    }else{
                        // reallocate *sigh*
                        if(filtSizeOK){
                            value_ptr v(new value_type(r0.shape));
                            convolve2d(*v, p0.value.cdata(), p1.value.cdata(), 0,1,m_nGroups);
                            r0.push(v);
                        }else{
                            // Alex' code has some serious restrictions; the one hurting me most
                            // is about the number of output maps (n%16==0).
                            // I'll emulate a less restricted version at some expense here
                            // by creating larger arrays if necessary>
                            unsigned int nFiltReal = r0.shape[0];
                            unsigned int nFiltTmp  = 16 * ceil(nFiltReal / 16.);                            // create intermediate representation of the outputs
                            value_type v(extents[nFiltTmp][r0.shape[1]][r0.shape[2]]);

                            // create intermediate copy of weights
                            value_type w(extents[p1.shape[0]][p1.shape[1]][nFiltTmp]);
                            w = 0.f;
                            //w[indices[index_range()][index_range()][index_range(0,nFiltTmp)]] = p1.value.cdata().copy();
                            tensor_view<float, cuv::dev_memory_space> wview(w,
                                    indices[index_range()][index_range()][index_range(0,nFiltReal)]);
                            wview = p1.value.cdata();

                            convolve2d(v, p0.value.cdata(), w, 0,1,m_nGroups);
                            value_ptr vp(new value_type(v[indices[index_range(0,nFiltReal)][index_range()][index_range()]]));
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
                void bprop(){
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

                    if(!filtSizeOK){
                        // create intermediate copy of deltas
                        tmp_r0delta.reset(new value_type(extents[nFiltTmp][r0.shape[1]][r0.shape[2]]));
                        {
                            *tmp_r0delta = 0.f;
                            (*tmp_r0delta)[indices[index_range(0,nFiltReal)][index_range()][index_range()]] = r0.delta.cdata();
                        }

                        // create intermediate copy of weights
                        tmp_w.reset(new value_type(extents[p1.shape[0]][p1.shape[1]][nFiltTmp]));
                        {
                            *tmp_w = 0.f;
                            (*tmp_w)[indices[index_range()][index_range()][index_range(0,nFiltReal)]] = p1.value.cdata().copy();
                        }

                        // create intermediate representation of filter derivative
                        tmp_dw.reset(new value_type(extents[p1.shape[0]][p1.shape[1]][nFiltTmp]));
                    }

                    if(p1.need_derivative){
                        // calculate p1 first, then we don't need activations
                        // anymore and can overwrite them. They are usually
                        // larger than the weights, so it should be better in this order.
                        const value_type& delta = r0.delta.cdata();
                        const value_type& img   = p0.value.cdata();
                        if(filtSizeOK && p1.can_overwrite_directly()){
                            if(filtSizeOK)
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, 0, 1, m_nGroups,m_partial_sum);
                            else
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, 0, 1, m_nGroups,m_partial_sum);
                        }
                        else if(filtSizeOK && p1.can_add_directly()){
                            if(filtSizeOK)
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, 0, 1, m_nGroups,m_partial_sum, 1.f,1.f);
                            else
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, 0, 1, m_nGroups,m_partial_sum, 1.f,1.f);
                        }
                        else{
                            if(filtSizeOK){
                                value_ptr ptr(new value_type(p1.shape));
                                value_type& dflt = *ptr;
                                d_conv2d_dfilt(dflt,delta,img, 0, 1, m_nGroups,m_partial_sum);
                                p1.push(ptr);
                            }else{
                                value_type& dflt = *tmp_dw;
                                d_conv2d_dfilt(dflt,*tmp_r0delta,img, 0, 1, m_nGroups,m_partial_sum);
                                value_ptr ptr(new value_type((*tmp_dw)[indices[index_range()][index_range()][index_range(0,nFiltReal)]].copy()));
                                p1.push(ptr);
                            }
                        }
                    }
                    if(p0.need_derivative){
                        // derivative w.r.t. images
                        const value_type& delta = r0.delta.cdata();
                        const value_type& flt   = p1.value.cdata();
                        if(filtSizeOK && p0.can_overwrite_directly()){
                            d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, 0, 1, m_nGroups);
                        }
                        else if (filtSizeOK && p0.can_add_directly()){
                            d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, 0, 1, m_nGroups,  1.f,1.f);
                        }
                        else{
                            if(filtSizeOK){
                                value_ptr ptr = p0.value;
                                p0.value.reset();       // try to overwrite input activations
                                value_type& v = ptr.data_onlyshape();
                                d_conv2d_dimg(v, delta, flt, 0,1,m_nGroups);
                                p0.push(ptr);
                            }else{
                                value_ptr ptr = p0.value;
                                p0.value.reset();       // try to overwrite input activations
                                value_type& v = ptr.data_onlyshape();
                                d_conv2d_dimg(v, *tmp_r0delta, *tmp_w, 0,1,m_nGroups);
                                p0.push(ptr);
                            }
                        }
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
                    /*
                     *  dst       (nFilt, nModules, nImg)
                     *  img       (nImgChan, nImgPix, nImg)
                     *  filter    (nFiltChan, nFiltPix, nFilt)
                     */
                    assert(m_params[0]->shape.size()==3);
                    assert(m_params[1]->shape.size()==3);
                    std::vector<unsigned int> dst(3);
                    const std::vector<unsigned int>& img = m_params[0]->shape;
                    const std::vector<unsigned int>& flt = m_params[1]->shape;
                    unsigned int nFilt    = flt[2];
                    unsigned int nImgPixX = sqrt(img[1]);
                    assert(nImgPixX*nImgPixX==img[1]);
                    unsigned int nFltPixX = sqrt(flt[1]);
                    assert(nFltPixX*nFltPixX==flt[1]);
                    dst[0] = nFilt;
#define _7848SQR(X) ((X)*(X));
                    dst[1] = _7848SQR( nImgPixX+1-nFltPixX );
                    dst[2] = img[2];
                    m_results[0]->shape = dst;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_nGroups;
                        ar & m_partial_sum;
                    }
        };

    class LocalPooling
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                cuv::alex_conv::pool_type m_pooltype;
                unsigned int m_subsx, m_stridex;
                value_ptr m_result;
            public:
                LocalPooling() :Op(1,1){} // for serialization
                LocalPooling(result_t& images, cuv::alex_conv::pool_type pt)
                    :Op(1,1),
                    m_pooltype(pt),
                    m_subsx(2),
                    m_stridex(2)
                {
                    add_param(0,images);
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    unsigned int outx = sqrt(p0.shape[1])/m_subsx;
                    cuvAssert(r0.shape[1]==outx*outx);
                    if(r0.can_overwrite_directly()){
                        //int subsX, int startX, int strideX, int outputsX, pool_type pooler
                        local_pool(*r0.overwrite_or_add_value(),p0.value.cdata(),
                            m_subsx, 0, m_stridex, outx, m_pooltype);
                        if(m_pooltype == PT_MAX && p0.need_derivative)
                            m_result = r0.overwrite_or_add_value(); // save for bprop
                    }else{
                        // reallocate *sigh*
                        value_ptr v(new value_type(r0.shape));
                        local_pool(*v,p0.value.cdata(),
                            m_subsx, 0, m_stridex, outx, m_pooltype);
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
                void bprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    //void local_max_pool_grad(target, const images, const maxGrads,
                    //        const maxActs, int subsX, int startX, int strideX, float factNew=1.f, float factOld=0.f);

                    //void local_avg_pool_grad(target, const avgGrads,
                    //        int subsX, int startX, int strideX);

                    if(m_pooltype == PT_AVG){
                        if(p0.can_overwrite_directly()){
                            local_avg_pool_grad(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_subsx,0,m_stridex);
                        }else{
                            // try overwriting p0.
                            //value_ptr ptr = p0.value;
                            value_ptr ptr(new value_type(p0.shape));
                            value_type& v = ptr.data_onlyshape();
                            local_avg_pool_grad(v, r0.delta.cdata(), m_subsx,0,m_stridex);
                            p0.push(ptr);
                        }
                    }else if(m_pooltype == PT_MAX){
                        if(p0.can_overwrite_directly()){
                            local_max_pool_grad(*p0.overwrite_or_add_value(), p0.value.cdata(), r0.delta.cdata(), m_result.cdata(), m_subsx,0,m_stridex);
                        }else if(p0.can_add_directly()){
                            local_max_pool_grad(*p0.overwrite_or_add_value(), p0.value.cdata(), r0.delta.cdata(), m_result.cdata(), m_subsx,0,m_stridex, 1.f,1.f);
                        }else{
                            value_ptr ptr(new value_type(p0.shape));
                            value_type& v = ptr.data_onlyshape();
                            local_max_pool_grad(v, p0.value.cdata(), r0.delta.cdata(), m_result.cdata(), m_subsx,0,m_stridex);
                            p0.push(ptr);
                        }
                        p0.value.reset(); 
                        m_result.reset();
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
                    /*
                     * images    (numFilters, imgPixels, numImages)
                     * dst:      (numFilters, outputs, numImages)
                     */
                    assert(m_params[0]->shape.size()==3);
                    std::vector<unsigned int> img = m_params[0]->shape;
                    std::vector<unsigned int> dst(3);
                    dst[0] = img[0];
                    dst[1] = img[1] / (m_subsx * m_subsx);
                    dst[2] = img[2];
                    cuvAssert(m_subsx*m_subsx*dst[1] == img[1]);
                    m_results[0]->shape = dst;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_pooltype & m_subsx & m_stridex;
                    }
        };

    /**
     * converts the (more natural) memory order nImg x nChann x nPix to the
     * order required by Alex' convolution routines.
     * 
     * ...which is nChann x nPix x nImg
     *
     * In bprop(), it does the opposite operation. This
     * is quite cheap compared to convolution itself.
     * (less than 1% for large images)
     *
     */
    class ReorderForConv
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
            public:
                ReorderForConv() :Op(1,1){} // for serialization
                ReorderForConv(result_t& images)
                    :Op(1,1)
                {
                    add_param(0,images);
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    if(r0.can_overwrite_directly()){
                        reorder_for_conv(*r0.overwrite_or_add_value(), p0.value.cdata());
                    }else{
                        value_ptr v(new value_type(r0.shape));
                        reorder_for_conv(*v, p0.value.cdata());
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
                        reorder_from_conv(*p0.overwrite_or_add_value(),r0.delta.cdata());
                    }else{
                        value_ptr v(new value_type(p0.shape));
                        reorder_from_conv(*v,r0.delta.cdata());
                        p0.push(v);
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
                    assert(m_params[0]->shape.size()==3);
                    const std::vector<unsigned int>& img = m_params[0]->shape;
                    std::vector<unsigned int> dst(3);
                    dst[0] = img[1];
                    dst[1] = img[2];
                    dst[2] = img[0];
                    m_results[0]->shape = dst;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };

}
#endif /* __OP_CONVOLVE_HPP__ */
