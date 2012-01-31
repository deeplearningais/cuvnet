#ifndef __CUVNET_MNIST_HPP__
#     define __CUVNET_MNIST_HPP__
#include <fstream>
#include "dataset.hpp"

namespace cuvnet
{
    struct mnist_dataset : public dataset{
        mnist_dataset(const std::string& path){
            std::cout << "Reading MNIST dataset..."<<std::flush;
            std::ifstream ftraind(path + "/train-images.idx3-ubyte",std::ios::in | std::ios::binary); // image data
            std::ifstream ftrainl(path + "/train-labels.idx1-ubyte",std::ios::in | std::ios::binary); // label data
            std::ifstream ftestd (path + "/t10k-images.idx3-ubyte",std::ios::in | std::ios::binary); // image data
            std::ifstream ftestl (path + "/t10k-labels.idx1-ubyte",std::ios::in | std::ios::binary); // label data
            assert(ftraind.is_open());
            assert(ftrainl.is_open());
            assert(ftestd.is_open());
            assert(ftestl.is_open());

            char buf[16];
            ftraind.read(buf,16);
            ftrainl.read(buf, 8);
            ftestd.read(buf,16);
            ftestl.read(buf, 8);
            cuv::tensor<unsigned char,cuv::host_memory_space> traind(cuv::extents[60000][784]);
            cuv::tensor<unsigned char,cuv::host_memory_space> trainl(cuv::extents[60000]);
            cuv::tensor<unsigned char,cuv::host_memory_space> testd(cuv::extents[10000][784]);
            cuv::tensor<unsigned char,cuv::host_memory_space> testl(cuv::extents[10000]);
            ftraind.read((char*)traind.ptr(), traind.size()); assert(ftraind.good());
            ftrainl.read((char*)trainl.ptr(), trainl.size()); assert(ftrainl.good());
            ftestd.read((char*)testd.ptr(), testd.size());    assert(ftestd.good());
            ftestl.read((char*)testl.ptr(), testl.size());    assert(ftestl.good());

            train_data.resize(traind.shape());
            test_data.resize(testd.shape());
            convert(train_data , traind); // convert data type
            convert(test_data  , testd); // convert data type

            train_labels.resize(cuv::extents[60000][10]);
            test_labels.resize(cuv::extents[10000][10]);
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

            channels = 1;
            image_size = 28;
            std::cout << "done."<<std::endl;
        }
    };
}


#endif /* __CUVNET_MNIST_HPP__ */
