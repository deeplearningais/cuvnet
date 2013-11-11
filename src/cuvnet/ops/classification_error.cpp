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

        //const value_type& inp0 = p0.value.cdata();           // original
        //const value_type& inp1 = p1.value.cdata();           // original
        value_type& inp0 = p0.value.data();           // original
        value_type& inp1 = p1.value.data();           // original

        std::vector<unsigned int> org_shape = p0.shape;
        unsigned int dim_other_axes;
        unsigned int batch_size;
        if(m_axis == 0) { 
            dim_other_axes = std::accumulate(p0.shape.begin(), --p0.shape.end(), 1, std::multiplies<unsigned int>());
            inp0.reshape(dim_other_axes, p0.shape.back());
            inp1.reshape(dim_other_axes, p0.shape.back());
            batch_size = inp0.shape(0);
        } else {
            dim_other_axes = std::accumulate(p0.shape.rbegin(), --p0.shape.rend(), 1, std::multiplies<unsigned int>());
            inp0.reshape(p0.shape.front(), dim_other_axes);
            inp1.reshape(p0.shape.front(), dim_other_axes);
            batch_size = inp0.shape(1);
        }

        cuv::tensor<int,Op::value_type::memory_space_type> a1 ( batch_size );
        cuv::tensor<int,Op::value_type::memory_space_type> a2 ( batch_size );
        if(m_axis == 0) {
            cuv::reduce_to_col(a1, inp0,cuv::RF_ARGMAX);
            cuv::reduce_to_col(a2, inp1,cuv::RF_ARGMAX);
        } else {
            cuv::reduce_to_row(a1, inp0,cuv::RF_ARGMAX);
            cuv::reduce_to_row(a2, inp1,cuv::RF_ARGMAX);
        }

        a1 -= a2;
        int n_wrong = batch_size - cuv::count(a1,0);

        value_ptr res(new value_type(cuv::extents[1]));
        *res = n_wrong/(float)batch_size;

        r0.push(res);

        inp0.reshape(org_shape);
        inp1.reshape(org_shape);

        p0.value.reset(); // forget it
        p1.value.reset(); // forget it
    }

    void ClassificationLoss::bprop(){
        throw std::runtime_error("there is no derivative for the zero-one loss!");
    }

    void ClassificationLoss::_determine_shapes(){
        assert(m_params[0]->shape == m_params[1]->shape);
        cuvAssert(m_axis == 0 || m_axis == m_params[0]->shape.size() - 1);
        m_results[0]->shape = std::vector<unsigned int>(1,1);
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
