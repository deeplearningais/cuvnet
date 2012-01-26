#ifndef __OP_CONVOLVE_HPP__
#     define __OP_CONVOLVE_HPP__

#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
namespace cuvnet
{
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
                    if(r0.can_overwrite_directly()){
                        convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), 0,1,m_nGroups);
                    }else if(r0.can_add_directly()){
                        convolve2d(*r0.overwrite_or_add_value(), p0.value.cdata(), p1.value.cdata(), 0,1,m_nGroups, 1.f,1.f);
                    }else{
                        // reallocate *sigh*
                        value_ptr v(new value_type(r0.shape));
                        convolve2d(*v, p0.value.cdata(), p1.value.cdata(), 0,1,m_nGroups);
                        r0.push(v);
                    }
                    if(!p0.need_derivative) p1.value.reset();
                    if(!p1.need_derivative) p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    using namespace cuv::alex_conv;

                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    assert(p0.need_derivative || p1.need_derivative);

                    if(p1.need_derivative){
                        // calculate this first, then we don't need activations
                        // anymore and can overwrite them. They are usually
                        // larger, so better in this order.
                        const value_type& delta = r0.delta.cdata();
                        const value_type& img   = p0.value.cdata();
                        if(p1.can_overwrite_directly())
                            d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, 0, 1, m_nGroups,m_partial_sum);
                        else if(p1.can_add_directly())
                            d_conv2d_dfilt(*p1.overwrite_or_add_value(),delta,img, 0, 1, m_nGroups,m_partial_sum, 1.f,1.f);
                        else{
                            // try overwriting filter values (which does not work, but at least we get the shape)
                            value_type& dflt = p1.value.data_onlyshape();
                            d_conv2d_dfilt(dflt,delta,img, 0, 1, m_nGroups,m_partial_sum);
                            p1.push(p1.value);
                        }
                    }
                    if(p0.need_derivative){
                        // derivative w.r.t. images
                        const value_type& delta = r0.delta.cdata();
                        const value_type& flt   = p1.value.cdata();
                        if(p0.can_overwrite_directly())
                            d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, 0, 1, m_nGroups);
                        else if (p0.can_add_directly())
                            d_conv2d_dimg(*p0.overwrite_or_add_value(),delta,flt, 0, 1, m_nGroups,  1.f,1.f);
                        else{
                            value_ptr ptr = p0.value;
                            p0.value.reset();       // try to overwrite input activations
                            value_type& v = ptr.data_onlyshape();
                            d_conv2d_dimg(v, delta, flt, 0,1,m_nGroups);
                            p0.push(ptr);
                        }
                    }
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
                    std::vector<unsigned int> img = m_params[0]->shape;
                    std::vector<unsigned int> flt = m_params[1]->shape;
                    unsigned int nFilt    = flt[2];
                    unsigned int nImgPixX = sqrt(img[1]);
                    assert(nImgPixX*nImgPixX==img[1]);
                    unsigned int nFltPixX = sqrt(flt[1]);
                    assert(nFltPixX*nFltPixX==flt[1]);
                    dst[0] = nFilt;
#define _7848SQR(X) ((X)*(X));
                    dst[1] = _7848SQR(nImgPixX+1-nFltPixX);
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
}
#endif /* __OP_CONVOLVE_HPP__ */
