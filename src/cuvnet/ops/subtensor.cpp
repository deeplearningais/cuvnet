#include "subtensor.hpp"

namespace cuvnet
{

    template<class T>
    struct view_of{
        typedef cuv::tensor_view<typename T::value_type, typename T::memory_space_type, typename T::memory_layout_type> type;
    };

    void Subtensor::get_subtensor(value_type &v){
        for(unsigned int i = 0, dim=0; i < m_starts.size(); i++){
            switch(dim){
                case 0:
                    if(m_is_degen[i])
                        v = v[cuv::indices[m_starts_det[i]]];
                    else
                        v = v[cuv::indices[cuv::index_range(m_starts_det[i], m_ends_det[i])]];
                    break;
                case 1:
                    if(m_is_degen[i])
                        v = v[cuv::indices[cuv::index_range()][m_starts_det[i]]];
                    else
                        v = v[cuv::indices[cuv::index_range()][cuv::index_range(m_starts_det[i], m_ends_det[i])]];
                    break;
                case 2:
                    if(m_is_degen[i])
                        v = v[cuv::indices[cuv::index_range()][cuv::index_range()][m_starts_det[i]]];
                    else
                        v = v[cuv::indices[cuv::index_range()][cuv::index_range()][cuv::index_range(m_starts_det[i], m_ends_det[i])]];
                    break;
                case 3:
                    if(m_is_degen[i])
                        v = v[cuv::indices[cuv::index_range()][cuv::index_range()][cuv::index_range()][m_starts_det[i]]];
                    else
                        v = v[cuv::indices[cuv::index_range()][cuv::index_range()][cuv::index_range()][cuv::index_range(m_starts_det[i], m_ends_det[i])]];
                    break;
            }

            if(!m_is_degen[i])
            {
                dim ++;
            }
        } 
    }

    void Subtensor::fprop(){

        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        if(r0.can_overwrite_directly()){
            value_type v = p0.value.cdata();
            get_subtensor(v);
            if(m_copy){
                // enforce copying by casting LHS to view
                // (in numpy slightly easier as lhs[:] = v)
                r0.overwrite_or_add_value()->copy_memory(v, false, 0);
                // semantically equivalent:
                //r0.overwrite_or_add_value() = v.copy();
            }
            else{
                // much faster, but unsafe since memory could be shared w/o cow_ptr guards
                r0.overwrite_or_add_value() = v;
            }
        }
        else{
            value_type v = p0.value.cdata();
            get_subtensor(v);

            if(m_copy || !v.is_c_contiguous()){
                value_ptr vp = value_ptr(new value_type(v.copy())); 
                r0.push(vp);
            }else{
                value_ptr vp = value_ptr(new value_type(v)); 
                r0.push(vp);
            }
        }
        p0.value.reset();
    }

    void Subtensor::bprop(){
        using namespace cuv;
        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

        if(p0.can_overwrite_directly()){
            value_type v = *p0.overwrite_or_add_value();
            v = 0.f;  // in case the input tensor is larger than the output tensor
            get_subtensor(v);
            // needs cast because memory is copied instead of copying pointer
            v.copy_memory(r0.delta.cdata(), false, 0);
        }else if(p0.can_add_directly()){
            value_type v = *p0.overwrite_or_add_value();
            get_subtensor(v);
            if(v.is_c_contiguous())
                v += r0.delta.cdata();
            else{
                value_type v2 = v.copy();
                v2 += r0.delta.cdata();
                v.copy_memory(v2, false, 0);
            }
        }else{
            value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
            *v = 0.f;  // in case the input tensor is larger than the output tensor
            value_type w = *v;
            // needs cast because memory is copied instead of copying pointer
            get_subtensor(w);
            w.copy_memory(r0.delta.cdata(), false, 0);
            p0.push(v);
        }
        r0.delta.reset();
    }

    void Subtensor::_determine_shapes(){
        std::vector<unsigned int> p0 = m_params[0]->shape;
        assert(m_starts.size() < 5);

        //std::cout << "  " << std::endl;
        //std::cout << "  " << std::endl;
        std::vector<unsigned int> dst;
        for (unsigned int i = 0; i < m_starts.size(); ++i){
            int start  = m_starts[i];
            int finish = m_ends[i];
            if (start <0) start  += p0[i];
            if(finish == std::numeric_limits<int>::min()) finish = p0[i];
            if (finish<0) finish += p0[i];

            cuvAssert(start <= finish);

            m_starts_det[i] = start;
            m_ends_det[i] = finish;
            if(!m_is_degen[i]){
                dst.push_back(finish-start);
            }
            //std::cout <<" s " << start << " f " << finish << std::endl;
        }

        m_results[0]->shape = dst;
    }

}
