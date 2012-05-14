#ifndef __CUVNET_SWISSROLL_HPP__
#     define __CUVNET_SWISSROLL_HPP__
#include <fstream>
#include <iostream>
#include "dataset.hpp"
#include <boost/filesystem.hpp>

namespace cuvnet
{
    struct swissroll_dataset : public dataset{
        swissroll_dataset(const std::string& path){
            namespace fs = boost::filesystem;
            using namespace cuv;
            std::cout << "Reading SwissRoll dataset..."<<std::flush;
            std::string ptrain = (path + "/swissroll-data-2048x3.dat");
            unsigned int n_test = 1024;
            assert(fs::exists(ptrain));
            std::ifstream ftraind(ptrain.c_str(),std::ios::in | std::ios::binary); // image data
            assert(ftraind.is_open());

            cuv::tensor<float,cuv::host_memory_space> traind(cuv::extents[2048][3]);
            ftraind.read((char*)traind.ptr(), sizeof(float)*traind.size()); assert(ftraind.good());

            train_data = traind[indices[index_range(0,2048-n_test)][index_range()]];
            test_data  = traind[indices[index_range(2048-n_test,2048)][index_range()]];

            train_labels.resize(cuv::extents[2048-n_test]);
            test_labels.resize( cuv::extents[n_test]);
            train_labels = 0.f;
            test_labels  = 0.f;

            channels = 1;
            binary   = false;
            image_size = 5;// arbitrary. no images in ds!
            std::cout << "done."<<std::endl;
        }
    };
}


#endif /* __CUVNET_SWISSROLL_HPP__ */
