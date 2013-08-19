#ifndef __MODULES_COMMON_HPP__
#     define __MODULES_COMMON_HPP__

#include <cuv.hpp>
namespace cuvnet
{
    /**
     *  This is the tensor type used to communicate between \c Op s.
     *  You can choose it to be either on device or host, thereby
     *  determining whether the GPU is used -- or not.
     */
	typedef cuv::tensor<float,cuv::dev_memory_space> matrix;
	typedef cuv::tensor<float,cuv::host_memory_space> host_matrix;
}

#endif /* __MODULES_COMMON_HPP__ */
