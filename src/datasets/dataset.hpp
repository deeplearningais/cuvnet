#ifndef __CUVNET_DATASET_HPP__
#     define __CUVNET_DATASET_HPP__

#include <cuv/basics/tensor.hpp>

namespace cuvnet
{
    struct dataset
    {
        cuv::tensor<float,cuv::host_memory_space> train_data;
        cuv::tensor<float,cuv::host_memory_space> val_data;
        cuv::tensor<float,cuv::host_memory_space> test_data;

        cuv::tensor<int,cuv::host_memory_space> train_labels;
        cuv::tensor<int,cuv::host_memory_space> val_labels;
        cuv::tensor<int,cuv::host_memory_space> test_labels;

        int channels;
        int image_size;
    };

}
#endif /* __CUVNET_DATASET_HPP__ */
