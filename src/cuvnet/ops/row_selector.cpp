#include "row_selector.hpp"

namespace cuvnet
{
    void RowSelector::set_random(bool b){
        m_random = b;
        if(!m_random && m_row < 0)
            m_row = 0;
    }

    void RowSelector::fprop(){
        using namespace cuv;
        if(m_random)
            m_row = drand48() * m_params[0]->shape[0];
        for(unsigned int i=0; i<this->get_n_params();i++){
            param_t::element_type&  p = *m_params[i];
            result_t::element_type& r = *m_results[i];
            if(!r.need_result)
            {
                p.value.reset();
                continue;
            }
            value_ptr v;
            if(m_copy)
                v.reset( new value_type((*p.value)[indices[m_row][index_range()]].copy()));
            else
                v.reset( new tensor_view<float,matrix::memory_space_type>(indices[m_row][index_range()],*p.value) );
            r.push(v);

            if(!m_copy){
                // do not forget inputs/outputs, since we propagated a /view/
                // - we need to make sure that noone overwrites the inputs
                //   by writing to the view --> pretend we need our inputs
                // - we need to make sure that noone overwrites the outputs
                //   by reusing the inputs --> pretend we need our outputs
            }else{
                p.value.reset();
            }
        }
    }
    void RowSelector::bprop(){
        using namespace cuv;
        for (unsigned int i = 0; i < get_n_params(); ++i)
        {
            param_t::element_type&  p = *m_params[i];
            result_t::element_type& r = *m_results[i];
            if(!p.need_derivative){
                r.delta.reset();
                continue;
            }
            if(!r.need_result){
                // we did not calculate anything, have to bprop zeros now
                if(p.can_add_directly()){
                    // no need to add zeros!
                }else if(p.can_overwrite_directly()){
                    value_type& oav = p.overwrite_or_add_value().data();
                    oav = 0.f; // need to set everything else to 0, since we're only setting a slice
                }else{
                    // reallocate *sigh*
                    value_ptr v(new value_type(p.shape));
                    *v = 0.f;
                    p.push(v);
                }
                continue;
            }
            if(p.can_add_directly()){
                value_type& oav = p.overwrite_or_add_value().data();
                value_type  view = oav[indices[m_row][index_range()]];
                view += r.delta.cdata();
            }else if(p.can_overwrite_directly()){
                value_type& oav = p.overwrite_or_add_value().data();
                oav = 0.f; // need to set everything else to 0, since we're only setting a slice
                oav[indices[m_row][index_range()]] = r.delta; // set directly
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(p.shape));
                *v = 0.f;
                (*v)[indices[m_row][index_range()]] = r.delta; // set directly
                p.push(v);
            }

            // /now/ we can forget these
            p.value.reset(); 
            r.delta.reset();
        }
    }

    void RowSelector::_determine_shapes(){
        // TODO: determining shapes works for dim==1 and dim>2, but fprop/bprop does not!

        // all inputs must have same first dimension!
        cuvAssert(m_params[0]->shape.size()>=1);
        unsigned int nrows0 = m_params[0]->shape[0];

        for(unsigned int i=0; i<this->get_n_params();i++)
        {
            param_t::element_type&  p = *m_params[i];
            cuvAssert(p.shape[0]==nrows0);

            std::vector<unsigned int> shape(p.shape.size()-1);
            for(unsigned int d=1;d<p.shape.size();d++){
                shape[d-1]=p.shape[d];
            }
            if(shape.size()==0)
                shape.push_back(1);
            m_results[i]->shape = shape;
        }
    }
}
