#ifndef __CUVNET_HISOGRAM_HPP__
#     define __CUVNET_HISOGRAM_HPP__

#include <cuvnet/common.hpp>

namespace cuvnet
{
    
    struct histogram
    {
        float m_min, m_max;
        unsigned int m_n_bins;
        cuv::tensor<unsigned int, cuv::host_memory_space> m_bins;

        template<class T, class M>
        histogram(unsigned int n_bins, const cuv::tensor<T,M>& t)
        {
            m_min = cuv::minimum(t);
            m_max = cuv::maximum(t);
            m_n_bins = n_bins;
            m_bins.resize(m_bins);
            m_bins = 0u;
            cuv::tensor<T, cuv::host_memory_space> th(t); // make sure we're on host
            unsigned int s = t.size();
            T* tp = th.ptr();
            float range = m_max-m_min;
            for(unsigned int i=0; i<s; i++){
                m_bins[(unsigned int)( (((float)*tp++)-m_min)/range )]++;
            }
        }
    };
}
#endif /* __CUVNET_HISOGRAM_HPP__ */
