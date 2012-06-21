#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>
#include <boost/format.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <tools/preprocess.hpp>
#include <tools/data_source.hpp>

using namespace cuvnet;

TEST(data_source_test, folder_loader){
    folder_loader fl("/home/local/datasets/VOC2011/TrainVal/VOCdevkit/VOC2011/JPEGImages",false);
    filename_processor fp;

    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;
    std::vector<tens_t> v;
    fl.get(v, 16, &fp);
    std::vector<unsigned int> shape = v[0].shape();
    std::copy(shape.begin(),shape.end(),std::ostream_iterator<unsigned int>(std::cout,", "));
    for (int i = 0; i < 16; ++i)
    {
        using namespace cuv;
        tensor<float,cuv::host_memory_space> img = v[i];
        libs::cimg::save( img, boost::str(boost::format("pp_img%03d.png")%i));
    }
}

