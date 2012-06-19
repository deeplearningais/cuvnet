#ifndef __CUVNET_MSRC_DESC_HPP__
#     define __CUVNET_MSRC_DESC_HPP__
#include <fstream>
#include <iostream>
#include "dataset.hpp"

namespace cuvnet
{
    /**
     * MSRC descriptors (not available publically).
     * @ingroup datasets
     */
    struct msrc_desc_dataset : public dataset{
        cuv::tensor<int,cuv::host_memory_space> imagen; // image numbers (one for each descriptor in the training set)

        msrc_desc_dataset(const std::string& path){
            std::cout << "Reading MSRC descriptors dataset..."<<std::flush;
            std::ifstream ftraind((path + "/X_train.bin").c_str(),std::ios::in | std::ios::binary); // image data
            std::ifstream ftrainl((path + "/y_train.bin").c_str(),std::ios::in | std::ios::binary); // label data
            std::ifstream ftestd ((path + "/X_val.bin").c_str(),std::ios::in | std::ios::binary); // image data
            std::ifstream ftestl ((path + "/y_val.bin").c_str(),std::ios::in | std::ios::binary); // label data

            // multiple descriptors can be taken from the same image
            std::ifstream fimagen((path + "/indicators.bin").c_str(), std::ios::in | std::ios::binary); // image numbers

            cuvAssert(ftraind.is_open());
            cuvAssert(ftrainl.is_open());
            cuvAssert(ftestd.is_open());
            cuvAssert(ftestl.is_open());
            cuvAssert(fimagen.is_open());

            cuv::tensor<float,cuv::host_memory_space> traind(cuv::extents[8021][2600]);
            cuv::tensor<int,cuv::host_memory_space> trainl(cuv::extents[8021]);
            cuv::tensor<float,cuv::host_memory_space> testd(cuv::extents[1852][2600]);
            cuv::tensor<int,cuv::host_memory_space> testl(cuv::extents[1852]);
            imagen.resize(cuv::extents[8021]);
            ftraind.read((char*)traind.ptr(), sizeof(float)*traind.size()); cuvAssert(ftraind.good());
            ftrainl.read((char*)trainl.ptr(), sizeof(int)*trainl.size()); cuvAssert(ftrainl.good());
            ftestd.read((char*)testd.ptr(), sizeof(float)*testd.size());    cuvAssert(ftestd.good());
            ftestl.read((char*)testl.ptr(), sizeof(int)*testl.size());    cuvAssert(ftestl.good());
            fimagen.read((char*)imagen.ptr(), sizeof(int)*imagen.size());    cuvAssert(fimagen.good());

            train_data.resize(traind.shape());
            test_data.resize(testd.shape());
            convert(train_data , traind); // convert data type
            convert(test_data  , testd); // convert data type

            train_labels.resize(cuv::extents[8021][21]);
            test_labels.resize(cuv::extents[1852][21]);
            train_labels = 0.f;
            test_labels = 0.f;
            for (unsigned int i = 0; i < trainl.size(); ++i){
                train_labels(i, trainl[i]) = 1;
            }
            for (unsigned int i = 0; i < testl.size(); ++i){
                test_labels(i, testl[i]) = 1;
            }

            //train_data = train_data[cuv::indices[cuv::index_range(0,5000)][cuv::index_range()]];
            //train_labels = train_labels[cuv::indices[cuv::index_range(0,5000)][cuv::index_range()]];

            binary = false;
            channels = 1;
            image_size = 28;
            std::cout << "done."<<std::endl;
        }
    };
}


#endif /* __CUVNET_MSRC_DESC_HPP__ */
