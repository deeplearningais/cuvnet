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

        value_type inp0 = p0.value.cdata();
        value_type inp1 = p1.value.cdata();
        //value_type inp2;

        bool ignore = m_params.size() == 3 ? true : false;
        //if (ignore) inp2 = m_params[2]->value.cdata();

        std::vector<unsigned int> org_shape = p0.shape;
        unsigned int dim_other_axes;
        unsigned int batch_size;
        if(m_axis == 0) { 
            dim_other_axes = inp0.size() / inp0.shape(inp0.ndim()-1);
            inp0.reshape(dim_other_axes, inp0.shape(inp0.ndim()-1));
            inp1.reshape(dim_other_axes, inp0.shape(inp0.ndim()-1));
            //if (ignore) inp2.reshape(dim_other_axes, inp0.shape(inp0.ndim()-1));
            batch_size = inp0.shape(0);
        } else {
            dim_other_axes = inp0.size() / inp0.shape(0);
            inp0.reshape(inp0.shape(0), dim_other_axes);
            inp1.reshape(inp0.shape(0), dim_other_axes);
            //if (ignore) inp2.reshape(inp0.shape(0), dim_other_axes);
            batch_size = inp0.shape(1);
        }

        // Apply ignore mask if needed
        value_type& inp0src = inp0;
        //cuv::tensor<float,Op::value_type::memory_space_type> cf( batch_size ); 
        cuv::tensor<float,Op::value_type::memory_space_type> cf( 1 ); 
        if (ignore) {
            param_t::element_type& p2 = *m_params[2];
            value_type inp2 = p2.value.cdata();
            if (m_axis == 0)
                inp2.reshape(dim_other_axes, 1);
            else
                inp2.reshape(1, dim_other_axes);
            value_type& inp0ign = p0.value.data_onlyshape();
            inp0ign.reshape(inp0.shape(0), inp0.shape(1));

            // apply ignore mask
            cuv::matrix_op_vec(inp0ign, inp0, inp2, m_axis == 0 ? 0 : 1, BF_MULT);
            inp0src = inp0ign;

            // determine amount of ignored part
            // todo
            //for (unsigned int i = 0; i < batch_size; i++)
                //cuv::count(inp2[cuv::indices[0]], 0);
                //cuv::count(inp2[cuv::extents[0]], 0);
            //cf[0] =  cuv::count(inp2, (float) 0) / (inp2.shape[0]*inp2.shape[1]);
            int c = cuv::count(inp2, (float) 0);
            cf[0] = c / (inp2.shape(0)*inp2.shape(1));
        }

        cuv::tensor<int,Op::value_type::memory_space_type> a1 ( batch_size );
        cuv::tensor<int,Op::value_type::memory_space_type> a2 ( batch_size );
        if(m_axis == 0) {
            cuv::reduce_to_col(a1, inp0src,cuv::RF_ARGMAX);
            cuv::reduce_to_col(a2, inp1,cuv::RF_ARGMAX);
        } else {
            cuv::reduce_to_row(a1, inp0src,cuv::RF_ARGMAX);
            cuv::reduce_to_row(a2, inp1,cuv::RF_ARGMAX);
        }

        a1 -= a2;
        int n_wrong = batch_size - cuv::count(a1,0);

        value_ptr res(new value_type(cuv::extents[1], value_ptr::s_allocator));
        *res = n_wrong/(float)batch_size;
        *res /= (1 - cf[0]);

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
        cuvAssert(m_axis == 0 || m_axis == m_params[0]->shape.size() - 1);

        if (m_params.size() == 3) {
            cuvAssert(m_params[2]->shape[m_axis] == 1);
            for (unsigned int i = 1; i < m_params[0]->shape.size()-1; i++)
                cuvAssert(m_params[0]->shape[i] == m_params[2]->shape[i]);
            // todo check last remaining axis
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
