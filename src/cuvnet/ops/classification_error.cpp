#include "classification_error.hpp"

namespace cuvnet
{
    
    void ClassificationLoss::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "ClassificationLoss";
    }

    void ClassificationLoss::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];
        result_t::element_type& r0 = *m_results[0];

        value_type inp0 = p0.value.cdata(); // estimator
        value_type inp1 = p1.value.cdata(); // teacher

        if(m_no_axis){
            value_ptr res(new value_type(cuv::extents[1], value_ptr::s_allocator));
            tensor<unsigned char, value_type::memory_space_type> t_estimator = inp0 > 0.f;
            tensor<unsigned char, value_type::memory_space_type> t_teacher = inp1 > 0.5f;
#if 1
            *res = 1.f - cuv::mean(t_estimator == t_teacher);
#else
            cuv::apply_binary_functor(t_estimator, t_estimator, t_teacher, cuv::BF_EQ);
            value_type col(extents[inp0.shape(0)], value_ptr::s_allocator);
            reduce_to_col(col, t_estimator);
            cuv::apply_scalar_functor(col, cuv::SF_LT, (float)inp0.shape(1) - 0.1f); // error when NOT ALL in that line had to be correct!
            *res = cuv::mean(col); // mean over batch
#endif
            r0.push(res);
            p0.value.reset(); // forget it
            p1.value.reset(); // forget it
            return;
        }

        bool ignore = m_params.size() == 3 ? true : false;

        std::vector<unsigned int> org_shape = p0.shape;
        unsigned int dim_other_axes;
        unsigned int batch_size;
        if(m_axis == 0) { 
            dim_other_axes = inp0.size() / inp0.shape(inp0.ndim()-1);
            inp0.reshape(dim_other_axes, inp0.shape(inp0.ndim()-1));
            inp1.reshape(dim_other_axes, inp0.shape(inp0.ndim()-1));
            batch_size = inp0.shape(0);
        } else {
            dim_other_axes = inp0.size() / inp0.shape(0);
            inp0.reshape(inp0.shape(0), dim_other_axes);
            inp1.reshape(inp0.shape(0), dim_other_axes);
            batch_size = inp0.shape(1);
        }

        // Apply ignore mask if needed
        value_type& inp0src = inp0;
        float avg_ign = 0;
        matrix a_ign ( batch_size, cuvnet::get_global_allocator() );
        if (ignore) {
            param_t::element_type& p2 = *m_params[2];
            value_type inp2 = p2.value.cdata();

            // reduce ignore term over classes.
            // TODO the ignore input should NOT have multiple classes!!!
            if (m_axis == 0) inp2.reshape(dim_other_axes, 1);
            else             inp2.reshape(1, dim_other_axes);
            if(m_axis == 0)  cuv::reduce_to_col(a_ign, inp2, cuv::RF_MEAN);
            else             cuv::reduce_to_row(a_ign, inp2, cuv::RF_MEAN);

            // determine amount of ignored part
            avg_ign = cuv::mean(a_ign);
        }

        matrix a1 ( batch_size, cuvnet::get_global_allocator() );
        matrix a2 ( batch_size, cuvnet::get_global_allocator() );
        if(m_axis == 0) {
            cuv::reduce_to_col(a1, inp0src,cuv::RF_ARGMAX);
            cuv::reduce_to_col(a2, inp1,cuv::RF_ARGMAX);
        } else {
            cuv::reduce_to_row(a1, inp0src,cuv::RF_ARGMAX);
            cuv::reduce_to_row(a2, inp1,cuv::RF_ARGMAX);
        }

        a1 -= a2;
        cuv::apply_scalar_functor(a1, cuv::SF_ABS);
        cuv::apply_scalar_functor(a1, cuv::SF_GT, 0.5f);
        if(ignore)
            a1 *= a_ign;

        float n_wrong = cuv::sum(a1);

        value_ptr res(new value_type(cuv::extents[1], value_ptr::s_allocator));
        if(!ignore || avg_ign == 0)
            *res = n_wrong/(float)batch_size;
        else
            *res = n_wrong / (float)batch_size / avg_ign;

        r0.push(res);

        p0.value.reset(); // forget it
        p1.value.reset(); // forget it
    }

    void ClassificationLoss::bprop(){
        throw std::runtime_error("there is no derivative for the zero-one loss!");
    }

    void ClassificationLoss::_determine_shapes(){
        m_results[0]->shape = std::vector<unsigned int>(1,1);
        if(!m_results[0]->need_result)
            return;
        assert(m_params[0]->shape == m_params[1]->shape);
        if(!m_no_axis){
            cuvAssert(m_axis == 0 || m_axis == m_params[0]->shape.size() - 1);

            if (m_params.size() == 3) {
                cuvAssert(m_params[2]->shape[m_axis] == 1);
                for (unsigned int i = 1; i < m_params[0]->shape.size()-1; i++)
                    cuvAssert(m_params[0]->shape[i] == m_params[2]->shape[i]);
                // todo check last remaining axis
            }
        }
    }


    /***************************************************
     * F2Measure
     ***************************************************/
    void F2Measure::fprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        param_t::element_type&  p1 = *m_params[1];

        const value_type& tch = p0.value.cdata();           // original
        const value_type& res = p1.value.cdata();           // original

        cuv::tensor<unsigned char, Op::value_type::memory_space_type> vtch (tch.shape());
        cuv::tensor<unsigned char, Op::value_type::memory_space_type> vres (tch.shape());

        vres = res > m_thresh_res;
        vtch = tch > m_thresh_tch;
        float tp, tn, fp, fn;

        if(m_params.size()==2){
            // no `ignore' mask
            tp = cuv::count( vres &&  vtch, (unsigned char)1);
            tn = cuv::count( vres ||  vtch, (unsigned char)0);
            fp = cuv::count( vres && !vtch, (unsigned char)1);
            fn = res.size() - (tp+tn+fp);
        }
        else{
            // with `ignore' mask
            param_t::element_type&  p2 = *m_params[2];
            const value_type& ign = p2.value.cdata();
            cuv::tensor<unsigned char, Op::value_type::memory_space_type> vign (tch.shape());
            vign = ign > 0.01f;

            tp = cuv::count( vres &&  vtch && vign, (unsigned char)1);
            tn = cuv::count( (vres ||  vtch) && vign, (unsigned char)0);
            fp = cuv::count( (vres && !vtch) && vign, (unsigned char)1);
            fn = cuv::count( (!vres && vtch) && vign, (unsigned char)1);
        }

        float precision = tp / (tp + fp);
        float recall    = tp / (tp + fn);
        float beta = 2;  // >2 weighs recall higher than precision
        float f2 = (1+beta*beta) * precision * recall / ( beta*beta*precision + recall );
        if(m_results[0]->can_overwrite_directly()){
            m_results[0]->overwrite_or_add_value().data() = f2;
        }else{
            value_ptr t_f2( new cuv::tensor<float, matrix::memory_space_type> (cuv::extents[1]) );
            (*t_f2)[0] = f2;
            m_results[0]->push(t_f2);
        }
        if(1 || m_results[1]->need_result){
            value_ptr t_tp( new cuv::tensor<float, matrix::memory_space_type> (cuv::extents[1]) );
            value_ptr t_tn( new cuv::tensor<float, matrix::memory_space_type> (cuv::extents[1]) );
            value_ptr t_fp( new cuv::tensor<float, matrix::memory_space_type> (cuv::extents[1]) );
            value_ptr t_fn( new cuv::tensor<float, matrix::memory_space_type> (cuv::extents[1]) );
            (*t_tp)[0] = tp;
            (*t_tn)[0] = tn;
            (*t_fp)[0] = fp;
            (*t_fn)[0] = fn;

            m_results[1]->push(t_tp);
            m_results[2]->push(t_tn);
            m_results[3]->push(t_fp);
            m_results[4]->push(t_fn);
        }

        p0.value.reset(); // forget it
        p1.value.reset(); // forget it
    }

    void F2Measure::bprop(){
        throw std::runtime_error("there is no derivative for the zero-one loss!");
    }

    void F2Measure::_determine_shapes(){
        assert(m_params[0]->shape == m_params[1]->shape);
        if(m_params.size() == 3){
            assert(m_params[1]->shape == m_params[2]->shape);
        }
        m_results[0]->shape = std::vector<unsigned int>(1,1);
        m_results[1]->shape = std::vector<unsigned int>(1,1);
        m_results[2]->shape = std::vector<unsigned int>(1,1);
        m_results[3]->shape = std::vector<unsigned int>(1,1);
        m_results[4]->shape = std::vector<unsigned int>(1,1);
    }

    void F2Measure::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.label = "F2Measure";
    }
}
