#include "mat_plus_vec.hpp"

namespace cuvnet
{
    void MatPlusVec::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            r0.overwrite_or_add_value() = p0.value;
            p0.value.reset(); // try to overwrite p0
            cuv::matrix_op_vec(*r0.overwrite_or_add_value(),*r0.overwrite_or_add_value(), p1.value.cdata(), m_axis, BF_ADD);
        }
        else if(r0.can_add_directly()){
            *r0.overwrite_or_add_value() += p0.value.cdata();
            cuv::matrix_op_vec(*r0.overwrite_or_add_value(),*r0.overwrite_or_add_value(), p1.value.cdata(), m_axis, BF_ADD, 1.f, 1.f);
        }else{
            // reallocate *sigh*
            value_ptr v = p0.value;
            p0.value.reset(); // try to overwrite p0
            cuv::matrix_op_vec(*v,*v, p1.value.cdata(), m_axis, BF_ADD);
            r0.push(v);
        }
        p0.value.reset();
        p1.value.reset();
    }

    void MatPlusVec::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        if(p0.need_derivative)
            p0.push(r0.delta);
        if(p1.need_derivative){
            unsigned int size = 1;
            unsigned int ndim = r0.shape.size();
            unsigned int rows = 1;
            for(unsigned int i = 0; i < ndim;i++){
                size *= r0.shape[i];
                if(i > m_axis)
                    rows *= r0.shape[i];
            }
            unsigned int cols = size / rows;
            if(p1.can_overwrite_directly()){
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*p1.overwrite_or_add_value(),
                            r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                else if (m_axis == 0)
                    reduce_to_col(*p1.overwrite_or_add_value(),
                            r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                else{
                    value_type v(cols);
                    value_type r = r0.delta.cdata();
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*p1.overwrite_or_add_value(), v, RF_ADD, 1.f, 0.f);
                }
            }
            else if(p1.can_add_directly()){
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*p1.overwrite_or_add_value(),
                            r0.delta.cdata(),RF_ADD, 1.f, 1.f);
                else if (m_axis == 0)
                    reduce_to_col(*p1.overwrite_or_add_value(),
                            r0.delta.cdata(),RF_ADD, 1.f, 1.f);
                else{
                    value_type v(cols);
                    value_type r = r0.delta.cdata();
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*p1.overwrite_or_add_value(), v, RF_ADD, 1.f, 1.f);
                }
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(p1.shape));
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*v, r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                else if(m_axis == 0)
                    reduce_to_col(*v, r0.delta.cdata(),RF_ADD, 1.f, 0.f);
                else{
                    value_type w(cols);
                    value_type r = r0.delta.cdata();
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(w, r,RF_ADD, 1.f, 0.f);
                    w.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*v, w, RF_ADD, 1.f, 0.f);
                }
                p1.push(v);
            }
        }
        r0.delta.reset();
    }

    void MatPlusVec::_determine_shapes(){
        assert(m_params[0]->shape.size()>=2);
        assert(m_params[1]->shape.size()==1);
        assert(m_params[0]->shape[m_axis] == m_params[1]->shape[0]);
        m_results[0]->shape = m_params[0]->shape;
    }

    void MatPlusVec::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "M + v, ax=" + boost::lexical_cast<std::string>(m_axis);
    }
    void MatTimesVec::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "M * v, ax=" + boost::lexical_cast<std::string>(m_axis);
    }
    void MatDivideVec::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "M / v, ax=" + boost::lexical_cast<std::string>(m_axis);
    }
    void MatTimesVec::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            r0.overwrite_or_add_value() = p0.value;
            if(!p0.need_derivative)
                p0.value.reset();   // avoid copy if it is not needed anymore
            cuv::matrix_op_vec(*r0.overwrite_or_add_value(),*r0.overwrite_or_add_value(), p1.value.cdata(), m_axis, BF_MULT);
        }
        else if(r0.can_add_directly()){
            *r0.overwrite_or_add_value() += p0.value.cdata();
            cuv::matrix_op_vec(*r0.overwrite_or_add_value(),*r0.overwrite_or_add_value(), p1.value.cdata(), m_axis, BF_MULT, 1.f, 1.f);
        }else{
            // reallocate *sigh*
            value_ptr v = p0.value;
            if(!p1.need_derivative)
                p0.value.reset();   // avoid copy if it is not needed anymore
            cuv::matrix_op_vec(*v,*v, p1.value.cdata(), m_axis, BF_MULT);
            r0.push(v);
        }
        if(!p1.need_derivative)
           p0.value.reset();
        if(!p0.need_derivative)
           p1.value.reset();
    }

    void MatTimesVec::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        if(p0.need_derivative){
            // p0.delta = matrix_times_{row/col} ( r.delta, v )
            if(!p1.need_derivative){
                // try overwriting r0.delta
                value_ptr& v = r0.delta;
                matrix_op_vec(v.data(), v.data(), p1.value.cdata(), m_axis, BF_MULT);
                p0.push(v);
            }else{
                // cannot overwrite r0, we need it later
                value_ptr  v(new value_type(r0.delta.cdata().copy()));
                matrix_op_vec(v.data(), v.data(), p1.value.cdata(), m_axis, BF_MULT);
                p0.push(v);
            }
        }
        if(p1.need_derivative){
            // try overwriting r0.delta
            const value_type& r0delta = r0.delta.cdata(); // remember true value of r0.delta
            value_type& m = r0.delta.data_onlyshape();    // this /may/ be the same as r0delta
            apply_binary_functor(m, r0delta, p0.value.cdata(), BF_MULT);

            unsigned int size = 1;
            unsigned int ndim = r0.shape.size();
            unsigned int rows = 1;
            for(unsigned int i = 0; i < ndim;i++){
                size *= r0.shape[i];
                if(i > m_axis)
                    rows *= r0.shape[i];
            }
            unsigned int cols = size / rows;
            if(p1.can_overwrite_directly()){
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 0.f);
                else if (m_axis == 0)
                    reduce_to_col(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 0.f);
                else{
                    value_type v(cols);
                    value_type r = m;
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*p1.overwrite_or_add_value(), v, RF_ADD, 1.f, 0.f);
                }
            }
            else if(p1.can_add_directly()){
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 1.f);
                else if (m_axis == 0)
                    reduce_to_col(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 1.f);
                else{
                    value_type v(cols);
                    value_type r = m;
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*p1.overwrite_or_add_value(), v, RF_ADD, 1.f, 1.f);
                }
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(p1.shape));
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*v, m,RF_ADD, 1.f, 0.f);
                else if(m_axis == 0)
                    reduce_to_col(*v, m,RF_ADD, 1.f, 0.f);
                else{
                    value_type w(cols);
                    value_type r = m;
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(w, r,RF_ADD, 1.f, 0.f);
                    w.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*v, w, RF_ADD, 1.f, 0.f);
                }
                p1.push(v);
            }
        }
        r0.delta.reset();
        p0.value.reset();
        p1.value.reset();
    }

    void MatTimesVec::_determine_shapes(){
        assert(m_params[0]->shape.size()>=2);
        assert(m_params[1]->shape.size()==1);
        assert(m_params[0]->shape[m_axis] == m_params[1]->shape[0]);
        m_results[0]->shape = m_params[0]->shape;
    }

    void MatDivideVec::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            r0.overwrite_or_add_value() = p0.value;
            if(!p0.need_derivative)
                p0.value.reset();   // avoid copy if it is not needed anymore
            cuv::matrix_op_vec(*r0.overwrite_or_add_value(),*r0.overwrite_or_add_value(), p1.value.cdata(), m_axis, BF_DIV);
        }
        else if(r0.can_add_directly()){
            *r0.overwrite_or_add_value() += p0.value.cdata();
            cuv::matrix_op_vec(*r0.overwrite_or_add_value(),*r0.overwrite_or_add_value(), p1.value.cdata(), m_axis, BF_DIV, 1.f, 1.f);
        }else{
            // reallocate *sigh*
            value_ptr v = p0.value;
            if(!p1.need_derivative)
                p0.value.reset();   // avoid copy if it is not needed anymore
            cuv::matrix_op_vec(*v,*v, p1.value.cdata(), m_axis, BF_DIV);
            r0.push(v);
        }
        if(!p1.need_derivative && !p0.need_derivative){
           p0.value.reset();
           p1.value.reset();
        }
        else if(!p1.need_derivative && p0.need_derivative)
           p0.value.reset();
        else if( p1.need_derivative && !p0.need_derivative)
        {}
    }

    void MatDivideVec::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        if(p0.need_derivative){
            // p0.delta = matrix_times_{row/col} ( r.delta, v )
            if(!p1.need_derivative){
                // try overwriting r0.delta
                value_ptr& v = r0.delta;
                matrix_op_vec(v.data(), v.data(), p1.value.cdata(), m_axis, BF_DIV);
                p0.push(v);
            }else{
                // cannot overwrite r0, we need it later
                value_ptr  v(new value_type(r0.delta.cdata().copy()));
                matrix_op_vec(v.data(), v.data(), p1.value.cdata(), m_axis, BF_DIV);
                p0.push(v);
            }
        }
        if(p1.need_derivative){
            // try overwriting r0.delta
            const value_type& r0delta = r0.delta.cdata(); // remember true value of r0.delta
            value_type& m = r0.delta.data_onlyshape();    // this /may/ be the same as r0delta

            // try overwriting p0.value
            value_type& v = p1.value.data();
            v *= v; // square v
            v *= -1.f;
            apply_binary_functor(m, r0delta, p0.value.cdata(), BF_MULT);
            matrix_op_vec(m, m, v, m_axis, BF_DIV);

            unsigned int size = 1;
            unsigned int ndim = r0.shape.size();
            unsigned int rows = 1;
            for(unsigned int i = 0; i < ndim;i++){
                size *= r0.shape[i];
                if(i > m_axis)
                    rows *= r0.shape[i];
            }
            unsigned int cols = size / rows;
            if(p1.can_overwrite_directly()){
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 0.f);
                else if (m_axis == 0)
                    reduce_to_col(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 0.f);
                else{
                    value_type v(cols);
                    value_type r = m;
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*p1.overwrite_or_add_value(), v, RF_ADD, 1.f, 0.f);
                }
            }
            else if(p1.can_add_directly()){
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 1.f);
                else if (m_axis == 0)
                    reduce_to_col(*p1.overwrite_or_add_value(),
                            m,RF_ADD, 1.f, 1.f);
                else{
                    value_type v(cols);
                    value_type r = m;
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(v, r,RF_ADD, 1.f, 0.f);
                    v.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*p1.overwrite_or_add_value(), v, RF_ADD, 1.f, 1.f);
                }
            }else{
                // reallocate *sigh*
                value_ptr v(new value_type(p1.shape));
                if(m_axis == p0.shape.size()-1)
                    reduce_to_row(*v, m,RF_ADD, 1.f, 0.f);
                else if(m_axis == 0)
                    reduce_to_col(*v, m,RF_ADD, 1.f, 0.f);
                else{
                    value_type w(cols);
                    value_type r = m;
                    r.reshape(cuv::extents[cols][rows]);
                    reduce_to_col(w, r,RF_ADD, 1.f, 0.f);
                    w.reshape(cuv::extents[cols/r0.shape[m_axis]][r0.shape[m_axis]]);
                    reduce_to_row(*v, w, RF_ADD, 1.f, 0.f);
                }
                p1.push(v);
            }
        }
        r0.delta.reset();
        p0.value.reset();
        p1.value.reset();
    }

    void MatDivideVec::_determine_shapes(){
        assert(m_params[0]->shape.size()>=2);
        assert(m_params[1]->shape.size()==1);
        assert(m_params[0]->shape[m_axis] == m_params[1]->shape[0]);
        m_results[0]->shape = m_params[0]->shape;
    }
}
