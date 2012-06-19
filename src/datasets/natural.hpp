#ifndef __CUVNET_NATURAL_HPP__
#     define __CUVNET_NATURAL_HPP__
#include <fstream>
#include <iostream>
#include "dataset.hpp"
#include <boost/filesystem.hpp>

namespace cuvnet
{
    /**
     * Classical van Hateren natural image patches dataset.
     *
     * There is a script \c extract_patches.py in the \c datasets/util directory that 
     * extracts patches from the van Hateren database.
     *
     * @ingroup datasets
     */
    struct natural_dataset : public dataset{
        natural_dataset(const std::string& path){
            namespace fs = boost::filesystem;
            std::cout << "Reading Natural Image Patches dataset..."<<std::flush;
            std::string ptrain = (path + "/patches_50000_16x16.bin");
            std::string ptest  = (path + "/patches_10000_16x16.bin");
            assert(fs::exists(ptrain));
            assert(fs::exists(ptest));
            std::ifstream ftraind(ptrain.c_str(),std::ios::in | std::ios::binary); // image data
            std::ifstream ftestd (ptest .c_str(),std::ios::in | std::ios::binary); // image data
            assert(ftraind.is_open());
            assert(ftestd.is_open());

            cuv::tensor<float,cuv::host_memory_space> traind(cuv::extents[50000][16*16]);
            cuv::tensor<float,cuv::host_memory_space> testd(cuv::extents[10000][16*16]);
            ftraind.read((char*)traind.ptr(), sizeof(float)*traind.size()); assert(ftraind.good());
            ftestd.read((char*)testd.ptr(), sizeof(float)*testd.size());    assert(ftestd.good());

            train_data.resize(traind.shape());
            test_data.resize(testd.shape());
            convert(train_data , traind); // convert data type
            convert(test_data  , testd); // convert data type

            train_labels.resize(cuv::extents[50000][10]);
            test_labels.resize( cuv::extents[10000][10]);
            train_labels = 0.f;
            test_labels  = 0.f;

            channels = 1;
            binary   = false;
            image_size = 16;
            std::cout << "done."<<std::endl;
        }
    };
}


#endif /* __CUVNET_NATURAL_HPP__ */
