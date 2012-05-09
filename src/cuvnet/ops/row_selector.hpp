#ifndef __ROW_SELECTOR_HPP__
#     define __ROW_SELECTOR_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * select a row out of a matrix
     *
     * rows may be chosen randomly during fprop, if not supplied.
     * if multiple inputs are given, the same row is chosen for all of them.
     */
    class RowSelector
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                int  m_row; ///< the row to be selected
                bool m_random; ///< whether to choose row randomly in fprop
                bool m_copy; ///< if true, does not use view on inputs (might save some memory at cost of speed if input matrix is huge and not needed by other ops)

            public:
                RowSelector(){} /// for serialization
                RowSelector(result_t& p0, int row=-1, bool copy=false):Op(1,1),m_row(row),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                }
                RowSelector(result_t& p0, result_t& p1, int row=-1, bool copy=false):Op(2,2),m_row(row),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                    add_param(1,p1);
                }
                RowSelector(result_t& p0, result_t& p1, result_t& p2, int row=-1, bool copy=false):Op(3,3),m_row(row),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                    add_param(1,p1);
                    add_param(2,p2);
                }

                void fprop(){
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
                void bprop(){
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
                void _determine_shapes(){
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
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_row & m_random & m_copy;
                    }
        };


	
}

#endif /* __ROW_SELECTOR_HPP__ */
