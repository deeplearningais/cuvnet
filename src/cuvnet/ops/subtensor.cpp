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
                r0.overwrite_or_add_value() = v.copy();
            }
            else{
                r0.overwrite_or_add_value() = v;
            }
        }else{
            value_ptr v;
            if(m_copy){
                v = value_ptr(new value_type(p0.value.cdata().copy())); // this safer but slower
            }
            else{
                v = value_ptr(new value_type(p0.value.cdata())); // this safer but slower
            }
            get_subtensor(*v);
            r0.push(v);
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
            v = 0.f;
            get_subtensor(v);
            // needs cast because memory is copied instead of copying pointer
            *static_cast<view_of<value_type>::type*>(&v) = r0.delta.cdata();
        }else if(p0.can_add_directly()){
            value_type v = *p0.overwrite_or_add_value();
            get_subtensor(v);
            // needs cast because memory is copied instead of copying pointer
            v += r0.delta.cdata();
        }else{
            value_ptr v(new value_type(p0.shape));
            *v = 0.f;
            value_type w = *v;
            // needs cast because memory is copied instead of copying pointer
            get_subtensor(w);
            *static_cast<view_of<value_type>::type*>(&w) = r0.delta.cdata();
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
