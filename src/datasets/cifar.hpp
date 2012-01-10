#ifndef __CIFAR_HPP__
#     define __CIFAR_HPP__

#include "dataset.hpp"
#include <cuv/basics/tensor.hpp>

namespace cuvnet
{

    struct cifar_dataset : dataset
    {
        cifar_dataset(){
            std::cout << "Reading CIFAR10 dataset..."<<std::flush;
            const unsigned int size = 3*32*32;
            train_data = cuv::tensor<float,cuv::host_memory_space>(cuv::extents[40000][size]);
            val_data   = cuv::tensor<float,cuv::host_memory_space>(cuv::extents[10000][size]);
            test_data  = cuv::tensor<float,cuv::host_memory_space>(cuv::extents[10000][size]);
            train_labels = cuv::tensor<int,cuv::host_memory_space>(cuv::extents[40000]);
            val_labels   = cuv::tensor<int,cuv::host_memory_space>(cuv::extents[10000]);
            test_labels  = cuv::tensor<int,cuv::host_memory_space>(cuv::extents[10000]);

            const char* datadir = "/home/local/datasets/CIFAR10/data_batch_%d.bin";
            char filename[255];
            unsigned char img[size];

            float* dest = train_data.ptr();
            for(unsigned int i=0;i<4;i++){
                sprintf(filename, datadir, i+1);
                std::ifstream ifs(filename, std::ios::in | std::ios::binary);
                for(unsigned int j=0;j<10000;j++){
                    train_labels[i*10000+j] = (int) ifs.get();
                    ifs.read((char*)img, size);
                    dest = std::copy(img,img+size, dest);
                }
            }
            {
                dest = val_data.ptr();
                sprintf(filename, datadir, 5);
                std::ifstream ifs(filename, std::ios::binary);
                for(unsigned int j=0;j<10000;j++){
                    val_labels[j] = (int) ifs.get();
                    ifs.read((char*)img, size);
                    dest = std::copy(img,img+size,dest);
                }
            }
            {
                dest = test_data.ptr();
                std::ifstream ifs("/home/local/datasets/CIFAR10/test_batch.bin", std::ios::binary);
                for(unsigned int j=0;j<10000;j++){
                    test_labels[j] = (int) ifs.get();
                    ifs.read((char*)img, size);
                    dest = std::copy(img,img+size,dest);
                }
            }
            channels = 3;
            image_size = 32;
            std::cout << "done."<<std::endl;
        }
    };

}
#endif /* __CIFAR_HPP__ */
