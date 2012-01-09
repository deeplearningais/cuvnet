#ifndef __CUVNET_PREPROCESS_HPP__
#     define __CUVNET_PREPROCESS_HPP__

#include<cuv/basics/tensor.hpp>

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
