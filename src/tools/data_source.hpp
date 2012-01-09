#ifndef __CUVNET_DATA_SOURCE_HPP
#     define __CUVNET_DATA_SOURCE_HPP

#include <string>
#include <vector>

/*
 * @file data_source.hpp provides methods to acquire raw 
 * images as strings in RAM in a common interface,
 * which can then be pre-processed by classes
 * in @see preprocess.hpp
 */

namespace cuvnet
{
    namespace detail
    {
        struct file_descr{
            std::string        name;
            size_t             size;
            std::vector<char*> content;
        };
    }
    class folder_loader{
        private:
            std::vector<detail::file_descr> m_files;
            void scan(const std::string& path, bool recursive);
        public:
            folder_loader(const std::string& path, bool recursive);
    };
    
}

#endif /* __CUVNET_DATA_SOURCE_HPP */
