#ifndef __CUVNET_LDPC_HPP__
#     define __CUVNET_LDPC_HPP__
#include <fstream>
#include <iostream>
#include "dataset.hpp"
#include <boost/filesystem.hpp>

namespace cuvnet
{
    struct ldpc_dataset : public dataset{
        ldpc_dataset(const std::string& path){
            namespace fs = boost::filesystem;
            using namespace cuv;
            std::cout << "Reading LDPC dataset..."<<std::flush;
            std::string ptrain = (path + "/ds_32768x30_float32.bin");
            assert(fs::exists(ptrain));
            std::ifstream ftraind(ptrain.c_str(),std::ios::in | std::ios::binary); // image data
            assert(ftraind.is_open());

            cuv::tensor<float,cuv::host_memory_space> traind(cuv::extents[32768][15*2]);
            ftraind.read((char*)traind.ptr(), sizeof(float)*traind.size()); assert(ftraind.good());

            train_data = traind[indices[index_range(0,29568)][index_range()]];
            test_data  = traind[indices[index_range(29568,32768)][index_range()]];

            train_labels.resize(cuv::extents[29568][10]);
            test_labels.resize( cuv::extents[3200][10]);
            train_labels = 0.f;
            test_labels  = 0.f;

            channels = 1;
            binary   = true;
            image_size = 5;// arbitrary. no images in ds!
            std::cout << "done."<<std::endl;
        }
    };
}


#endif /* __CUVNET_LDPC_HPP__ */
