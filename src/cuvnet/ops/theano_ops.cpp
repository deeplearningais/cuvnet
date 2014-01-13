
#include "theano_ops.hpp"

namespace cuvnet
{

    void ShuffleDim::fprop(){
        using namespace cuv;
        using namespace cuv::theano_ops;
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
    }


    void ShuffleDim::bprop(){
        using namespace cuv;
        using namespace cuv::theano_ops;
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
    }


    void FlipDims::bprop(){
        using namespace cuv;
        using namespace cuv::theano_ops;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            flip_dims_vec(*p0.overwrite_or_add_value(), r0.delta.cdata(), m_pattern);
        }else{
            value_ptr res(new value_type(p0.value.cdata().shape()));
            value_type& dres = *res;
            flip_dims_vec(dres, r0.delta.cdata(), m_pattern);
            p0.push(res);
        }
        r0.delta.reset();
        p0.value.reset(); // now we don't need it anymore ;)
    }

    void FlipDims::_determine_shapes(){
       assert(m_params[0]->shape.size() == 4);
       m_results[0]->shape = m_params[0]->shape;
    }
}
