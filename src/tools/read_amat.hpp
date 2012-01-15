#ifndef __READ_AMAT_HPP__
#     define __READ_AMAT_HPP__
#include <string>
#include <cuv/basics/tensor.hpp>

namespace cuvnet
{
    void read_amat(cuv::tensor<float,cuv::host_memory_space>& t, const std::string& zipfile_name, const std::string& zippedfile_name);
    void read_amat_with_label(
            cuv::tensor<float,cuv::host_memory_space>& t,
            cuv::tensor<int,cuv::host_memory_space>& l,
            const std::string& zipfile_name, const std::string& zippedfile_name);
}

#endif /* __READ_AMAT_HPP__ */
