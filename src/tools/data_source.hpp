#ifndef __CUVNET_DATA_SOURCE_HPP
#     define __CUVNET_DATA_SOURCE_HPP

#include <string>
#include <vector>
#include "preprocess.hpp"

/*
 * @file data_source.hpp provides methods to acquire raw 
 * images as strings in RAM in a common interface,
 * which can then be pre-processed by classes
 * in @see preprocess.hpp
 */

namespace cuvnet
{
    struct folder_loader{
        public:
            std::vector<detail::file_descr> m_files;
            unsigned int m_idx;
            void scan(const std::string& path, bool recursive);
        public:
            folder_loader(const std::string& path, bool recursive);
            void get(std::vector<cuv::tensor<float,cuv::host_memory_space> >& res, const unsigned int n, preprocessor* pp);
    };
    
}

#endif /* __CUVNET_DATA_SOURCE_HPP */
