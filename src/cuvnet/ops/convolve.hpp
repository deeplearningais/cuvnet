#ifndef __OP_CONVOLVE_HPP__
#     define __OP_CONVOLVE_HPP__

#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
namespace cuvnet
{

    /**
     * convolve a set of images using a set of filters.
     *
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
     *  @ingroup Ops
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
                int m_padding_start;
            public:
                Convolve() :Op(2,1){} // for serialization

                /**
                 * constructor.
                 *
                 * @param images nChannels x nPixels x nImages
                 * @param filters nFiltChannels x nFiltPix x nFilt
                 * @param partial_sum optimization parameter of alex' convolution routines. Good values are probably 4 or 8.
                 */
                Convolve(result_t& images, result_t& filters, bool padding, unsigned int partial_sum=4)
                    :Op(2,1)
                    ,m_nGroups(1)
                    ,m_partial_sum(partial_sum)
                    ,m_padding_start(padding)
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
                        convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), m_padding_start,1,m_nGroups);
                    }else if(filtSizeOK && r0.can_add_directly()){
                        convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), m_padding_start,1,m_nGroups, 1.f,1.f);
                    }else{
                        // reallocate *sigh*
                        if(filtSizeOK){
                            value_ptr v(new value_type(r0.shape));
                            convolve2d(*v, p0.value.cdata(), p1.value.cdata(), m_padding_start,1,m_nGroups);
                            r0.push(v);
                        }else{
                            // Alex' code has some serious restrictions; the one hurting me most
                            // is about the number of output maps (n%16==0).
                            // I'll emulate a less restricted version at some expense here
                            // by creating larger arrays if necessary>
                            unsigned int nFiltReal = r0.shape[0];
                            unsigned int nFiltTmp  = 16 * ceil(nFiltReal / 16.);                            // create intermediate representation of the outputs
                            value_type tmp_dst(extents[nFiltTmp][r0.shape[1]][r0.shape[2]][r0.shape[3]]);

                            // create intermediate copy of weights
                            value_type tmp_flt(extents[p1.shape[0]][p1.shape[1]][nFiltTmp]);
                            tmp_flt = 0.f;
                            //tmp_flt[indices[index_range()][index_range()][index_range(0,nFiltTmp)]] = p1.value.cdata().copy();
                            tensor_view<float, cuv::dev_memory_space> wview(tmp_flt,
                                    indices[index_range()][index_range()][index_range(0,nFiltReal)]);
                            wview = p1.value.cdata();

                            convolve2d(tmp_dst, p0.value.cdata(), tmp_flt, m_padding_start,1,m_nGroups);
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
                        tmp_r0delta.reset(new value_type(extents[nFiltTmp][r0.shape[1]][r0.shape[2]][r0.shape[3]]));
                        {
                            *tmp_r0delta = 0.f;
                            (*tmp_r0delta)[indices[index_range(0,nFiltReal)][index_range()][index_range()][index_range()]] = r0.delta.cdata();
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
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, m_padding_start, 1, m_nGroups,m_partial_sum);
                            else
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, m_padding_start, 1, m_nGroups,m_partial_sum);
                        }
                        else if(filtSizeOK && p1.can_add_directly()){
                            if(filtSizeOK)
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, m_padding_start, 1, m_nGroups,m_partial_sum, 1.f,1.f);
                            else
                                d_conv2d_dfilt(*p1.overwrite_or_add_value(),*tmp_r0delta,img, m_padding_start, 1, m_nGroups,m_partial_sum, 1.f,1.f);
                        }
                        else{
                            if(filtSizeOK){
                                value_ptr ptr(new value_type(p1.shape));
                                value_type& dflt = *ptr;
                                d_conv2d_dfilt(dflt,delta,img, m_padding_start, 1, m_nGroups,m_partial_sum);
                                p1.push(ptr);
                            }else{
                                value_type& dflt = *tmp_dw;
                                d_conv2d_dfilt(dflt,*tmp_r0delta,img, m_padding_start, 1, m_nGroups,m_partial_sum);
                                value_ptr ptr(new value_type(dflt[indices[index_range()][index_range()][index_range(0,nFiltReal)]].copy()));
                                p1.push(ptr);
                            }
                        }
                    }
                    if(p0.need_derivative){
                        // derivative w.r.t. images
                        const value_type& delta = r0.delta.cdata();
                        const value_type& flt   = p1.value.cdata();
                        if(filtSizeOK && p0.can_overwrite_directly()){
                            d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, m_padding_start, 1, m_nGroups);
                        }
                        else if (filtSizeOK && p0.can_add_directly()){
                            d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, m_padding_start, 1, m_nGroups,  1.f,1.f);
                        }
                        else{
                            if(filtSizeOK){
                                value_ptr ptr = p0.value;
                                p0.value.reset();       // try to overwrite input activations
                                value_type& v = ptr.data_onlyshape();
                                d_conv2d_dimg(v, delta, flt, m_padding_start,1,m_nGroups);
                                p0.push(ptr);
                            }else{
                                value_ptr ptr = p0.value;
                                p0.value.reset();       // try to overwrite input activations
                                value_type& v = ptr.data_onlyshape();
                                d_conv2d_dimg(v, *tmp_r0delta, *tmp_w, m_padding_start,1,m_nGroups);
                                p0.push(ptr);
                            }
                        }
                    }
                    p0.value.reset();
                    p1.value.reset();
                    r0.delta.reset();
                }

                void _determine_shapes(){
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

                    if(m_padding_start)
                        m_padding_start = -(int)nFltPixX/2; // assume nFltPixX%2==1

                    unsigned int nOutPixX = m_padding_start 
                        ? (nImgPixX)
                        : (nImgPixX+1-nFltPixX);
                    unsigned int nOutPixY = m_padding_start 
                        ? (nImgPixY)
                        : (nImgPixY+1-nFltPixX);

                    dst[0] = nFilt;
                    dst[1] = nOutPixY;
                    dst[2] = nOutPixX;
                    dst[3] = img[3];
                    m_results[0]->shape = dst;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_nGroups;
                        ar & m_partial_sum;
                        ar & m_padding_start;
                    }
        };

    /**
     * Bed-of-nails subsampling (take every n-th value in the input maps)
     *
     * @ingroup Ops
     */
    class BedOfNails
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_startx, m_stridex;
            public:
                BedOfNails() :Op(1,1){} // for serialization
                BedOfNails(result_t& images, int stridex=2, int startx=0)
                    :Op(1,1),
                    m_startx(startx),
                    m_stridex(stridex)
                {
                    add_param(0,images);
                }
                void fprop(){
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
                        value_ptr v(new value_type(r0.shape));
                        bed_of_nails(*v, p0.value.cdata(), m_startx,m_stridex);
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
                        bed_of_nails_grad(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_startx,m_stridex);
                    }else if(p0.can_add_directly()){
                        bed_of_nails_grad(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_startx,m_stridex, 1.f, 1.f);
                    }else{
                        value_ptr ptr(new value_type(p0.shape));
                        bed_of_nails_grad(*ptr, r0.delta.cdata(), m_startx,m_stridex);
                        p0.push(ptr);
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
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

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_stridex & m_startx;
                    }
        };

    /**
     * Separable Filter.
     *
     * @ingroup Ops
     */
    class SeparableFilter
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                matrix m_kernel;
            public:
                SeparableFilter() :Op(1,1){} // for serialization
                SeparableFilter(result_t& images, const matrix& kernel)
                    :Op(1,1),
                    m_kernel(kernel)
                {
                    add_param(0,images);
                }
                void fprop(){
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
                void bprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    if(p0.can_overwrite_directly()){
                        value_type v(p0.shape);
                        cuv::alex_conv::gaussian_blur(v, r0.delta.cdata(), m_kernel, true);
                        cuv::alex_conv::gaussian_blur(*p0.overwrite_or_add_value(), v, m_kernel, false);
                    }else if(p0.can_add_directly()){
                        value_type v(p0.shape);
                        cuv::alex_conv::gaussian_blur(v, r0.delta.cdata(), m_kernel, true);
                        cuv::alex_conv::gaussian_blur(*p0.overwrite_or_add_value(), v, m_kernel, false, 1.f, 1.f);
                    }else{
                        // try to overwrite r0.delta
                        value_type v(p0.shape);
                        cuv::alex_conv::gaussian_blur(v, r0.delta.cdata(), m_kernel, true);
                        cuv::alex_conv::gaussian_blur(r0.delta.data(), v, m_kernel, false);
                        p0.push(r0.delta);
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
                    /*
                     * images    (numFilters, imgPixY, imgPixX, numImages)
                     * dst:      (numFilters, imgPixY, imgPixX, numImages)
                     */
                    assert(m_params[0]->shape.size()==4);
                    m_results[0]->shape = m_params[0]->shape;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_kernel;
                    }
        };

    /**
     * Maximum pooling or subsampling of image regions.
     *
     * @ingroup Ops
     */
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
                virtual void release_data(){
                    m_result.reset();
                    Op::release_data();
                }
                void fprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    unsigned int outy = p0.shape[1]/m_subsx;
                    unsigned int outx = p0.shape[2]/m_subsx;
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
                            value_type& v = *ptr;
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
                            value_type& v = *ptr;
                            local_max_pool_grad(v, p0.value.cdata(), r0.delta.cdata(), m_result.cdata(), m_subsx,0,m_stridex);
                            p0.push(ptr);
                        }
                        p0.value.reset(); 
                        m_result.reset();
                    }
                    p0.value.reset();
                    r0.delta.reset();
                    m_result.reset();
                }

                void _determine_shapes(){
                    /*
                     * images    (numFilters, imgPixels, numImages)
                     * dst:      (numFilters, outputs, numImages)
                     */
                    assert(m_params[0]->shape.size()==4);
                    std::vector<unsigned int> img = m_params[0]->shape;
                    std::vector<unsigned int> dst(4);
                    dst[0] = img[0];
                    dst[1] = img[1] / m_subsx;
                    dst[2] = img[2] / m_subsx;
                    dst[3] = img[3];
                    cuvAssert(img[1]==img[2]); // currently, cudaConv2 only supports square images for pooling
                    cuvAssert(m_subsx * dst[1] == img[1]);
                    cuvAssert(m_subsx * dst[2] == img[2]);
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
     * @ingroup Ops
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
                    assert(m_params[0]->shape.size()==4);
                    const std::vector<unsigned int>& img = m_params[0]->shape;
                    std::vector<unsigned int> dst(4);
                    dst[0] = img[1];
                    dst[1] = img[2];
                    dst[2] = img[3];
                    dst[3] = img[0];
                    m_results[0]->shape = dst;
                }

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };

    /**
     * Does the opposite of \c ReorderForConv.
     *
     * @ingroup Ops
     */
    class ReorderFromConv
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
            public:
                ReorderFromConv() :Op(1,1){} // for serialization
                ReorderFromConv(result_t& images)
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
                        reorder_from_conv(*r0.overwrite_or_add_value(), p0.value.cdata());
                    }else{
                        value_ptr v(new value_type(r0.shape));
                        reorder_from_conv(*v, p0.value.cdata());
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
                        reorder_for_conv(*p0.overwrite_or_add_value(),r0.delta.cdata());
                    }else{
                        value_ptr v(new value_type(p0.shape));
                        reorder_for_conv(*v,r0.delta.cdata());
                        p0.push(v);
                    }
                    r0.delta.reset();
                }

                void _determine_shapes(){
                    assert(m_params[0]->shape.size()==4);
                    const std::vector<unsigned int>& img = m_params[0]->shape;
                    std::vector<unsigned int> dst(4);
                    dst[0] = img[3];
                    dst[1] = img[0];
                    dst[2] = img[1];
                    dst[3] = img[2];
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
