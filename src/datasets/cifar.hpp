#ifndef __CIFAR_HPP__
#     define __CIFAR_HPP__

#include "dataset.hpp"
#include <cuv/basics/tensor.hpp>

namespace cuvnet
{

    /**
     * Access to the CIFAR-10 dataset by Alex Krizhevsky.
     * @ingroup datasets
     */
    struct cifar_dataset : dataset
    {
        cifar_dataset(const std::string& path){
            std::cout << "Reading CIFAR10 dataset..."<<std::flush;
            const unsigned int size = 3*32*32;
            train_data.resize(cuv::extents[50000][size]);
            //val_data.resize(cuv::extents[10000][size]);
            test_data.resize(cuv::extents[10000][size]);
            train_labels.resize(cuv::extents[50000]);
            //val_labels.resize(cuv::extents[10000]);
            test_labels.resize(cuv::extents[10000]);

            const char* templ = "%s/data_batch_%d.bin";
            char filename[255];
            unsigned char img[size];

            cuv::tensor<unsigned char,cuv::host_memory_space> trainl(50000);
            cuv::tensor<unsigned char,cuv::host_memory_space> testl(10000);

            float* dest = train_data.ptr();
            for(unsigned int i=0;i<5;i++){
                sprintf(filename, path.c_str(), templ, i+1);
                std::ifstream ifs(filename, std::ios::in | std::ios::binary);
                for(unsigned int j=0;j<10000;j++){
                    trainl[i*10000+j] = (int) ifs.get();
                    ifs.read((char*)img, size);
                    dest = std::copy(img,img+size, dest);
                }
            }
            /*
             *{
             *    dest = val_data.ptr();
             *    sprintf(filename, path.c_str(), templ, 5);
             *    std::ifstream ifs(filename, std::ios::binary);
             *    for(unsigned int j=0;j<10000;j++){
             *        val_labels[j] = (int) ifs.get();
             *        ifs.read((char*)img, size);
             *        dest = std::copy(img,img+size,dest);
             *    }
             *}
             */
            {
                dest = test_data.ptr();
                std::ifstream ifs("/home/local/datasets/CIFAR10/test_batch.bin", std::ios::binary);
                for(unsigned int j=0;j<10000;j++){
                    testl[j] = (int) ifs.get();
                    ifs.read((char*)img, size);
                    dest = std::copy(img,img+size,dest);
                }
            }

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

            channels = 3;
            binary   = false;
            image_size = 32;
            std::cout << "done."<<std::endl;
        }
    };

}
#endif /* __CIFAR_HPP__ */
