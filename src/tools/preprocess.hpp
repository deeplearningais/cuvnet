#ifndef __CUVNET_PREPROCESS_HPP__
#     define __CUVNET_PREPROCESS_HPP__

#include<cuv/basics/tensor.hpp>
#include<cuv/matrix_ops/matrix_ops.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>

namespace cuvnet
{
    namespace detail
    {
        struct file_descr{
            std::string        name;
            size_t             size;
            std::vector<char> content;
        };
    }

    template<class M=cuv::host_memory_space>
    class zero_mean_unit_variance{
        public:
            cuv::tensor<float, M> m_mean;
            cuv::tensor<float, M> m_std;
        public:
            void fit(const cuv::tensor<float,M>& train){
                using namespace cuv;
                m_mean.resize(extents[train.shape(1)]);
                m_std .resize(extents[train.shape(1)]);
                reduce_to_row(m_mean,train,RF_ADD);
                reduce_to_row(m_std, train,RF_ADD_SQUARED);
                m_std  /= (float)train.shape(0);
                m_mean /= (float)train.shape(0);
                m_std -= ::operator*(m_mean,m_mean);
                apply_scalar_functor(m_std, SF_SQRT);
                apply_scalar_functor(m_mean, SF_NEGATE);
            }
            void transform(cuv::tensor<float,M>& data){
                cuv::matrix_plus_row(data,m_mean); // mean is negated already
                cuv::matrix_divide_row(data,m_std);
            }
            void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
    };

    template<class M=cuv::host_memory_space>
    class global_min_max_normalize{
        private:
            float m_min, m_max;
            float m_add, m_fact;
        public:
            global_min_max_normalize(float min=0.f, float max=1.f):m_min(min),m_max(max){}
            void fit(const cuv::tensor<float,M>& train){
                float xmin = cuv::minimum(train); // minimum
                float xmax = cuv::maximum(train); // range
                m_add  = (xmin*m_max + xmax*m_min)/(xmax-xmin);
                m_fact = (m_max-m_min)/(xmax-xmin);
            }
            void transform(cuv::tensor<float,M>& data){
                data *= m_fact;
                data += m_add;
            }
            void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
    };
    class preprocessor{
        public:
            virtual void process_filestring(cuv::tensor<float,cuv::host_memory_space>& dst, const char* buf, size_t n)=0;
    };

    class patch_extractor
    : public preprocessor
    {
        private:
            unsigned int m_s0, m_s1;
        public:
            patch_extractor(unsigned int s0, unsigned int s1):m_s0(s0),m_s1(s1){}
            void process_filestring(cuv::tensor<float,cuv::host_memory_space>& dst, const char* buf, size_t n);
    };
}
#endif /* __CUVNET_PREPROCESS_HPP__ */
