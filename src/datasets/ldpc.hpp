#ifndef __CUVNET_LDPC_HPP__
#     define __CUVNET_LDPC_HPP__
#include <fstream>
#include <iostream>
#include "dataset.hpp"
#include <boost/filesystem.hpp>

namespace cuvnet
{
    /**
     * contains the low-density parity check-code dataset.
     *
     * There is an \c ldpc.cpp file in \c datasets/util that generates the
     * required dataset file.
     * 
     * @ingroup datasets
     */
    struct ldpc_dataset : public dataset{
        ldpc_dataset(const std::string& path){
            namespace fs = boost::filesystem;
            using namespace cuv;
            std::cout << "Reading LDPC dataset..."<<std::flush;
            std::string ptrain = (path + "/ds_32768x30_float32.bin");
            //std::string ptrain = (path + "/ds_60000x36_float32.bin");
            unsigned int n_test = 10000;
            assert(fs::exists(ptrain));
            std::ifstream ftraind(ptrain.c_str(),std::ios::in | std::ios::binary); // image data
            assert(ftraind.is_open());

            cuv::tensor<float,cuv::host_memory_space> traind(cuv::extents[32768][15*2]);
            ftraind.read((char*)traind.ptr(), sizeof(float)*traind.size()); assert(ftraind.good());

            train_data = traind[indices[index_range(0,32768-n_test)][index_range()]].copy();
            test_data  = traind[indices[index_range(32768-n_test,32768)][index_range()]].copy();

            //train_labels.resize(cuv::extents[32768-n_test][15]);
            //test_labels.resize( cuv::extents[n_test][15]);
            train_labels = train_data[indices[index_range()][index_range(0,15)]].copy();
            test_labels  = test_data[indices[index_range()][index_range(0,15)]].copy();

            channels = 1;
            binary   = true;
            image_size = 5;// arbitrary. no images in ds!
            std::cout << "done."<<std::endl;
        }
    };
}


#endif /* __CUVNET_LDPC_HPP__ */
