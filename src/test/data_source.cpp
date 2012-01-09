#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>

#include <cuv/basics/tensor.hpp>
#include <tools/preprocess.hpp>
#include <tools/data_source.hpp>

using namespace cuvnet;

TEST(data_source_test, folder_loader){
    folder_loader fl("/home/local/datasets/VOC2011/TrainVal/VOCdevkit/VOC2011/JPEGImages",false);
    patch_extractor pe(6,6);

    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;
    std::vector<tens_t> v;
    fl.get(v, 16, &pe);
}

