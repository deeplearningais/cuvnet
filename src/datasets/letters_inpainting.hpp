#ifndef __CUVNET_LETTERS_INPAINTING_HPP__
#     define __CUVNET_LETTERS_INPAINTING_HPP__
#include <fstream>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include<cuv.hpp>
#include "dataset.hpp"

namespace cuvnet
{
    /**
     * CVPR letters inpainting dataset
     * @ingroup datasets
     */
    struct letters_inpainting : public dataset{
        letters_inpainting(const std::string& path){
            std::cout << "Reading letters dataset..."<<std::flush;
            std::ifstream ftraind((path + "/images_train.bin").c_str(),std::ios::in | std::ios::binary); // image data
            std::ifstream ftrainl((path + "/labels_train.bin").c_str(),std::ios::in | std::ios::binary); // label data
            std::ifstream ftestd ((path + "/images_test.bin").c_str(),std::ios::in | std::ios::binary); // image data
            std::ifstream ftestl ((path + "/labels_test.bin").c_str(),std::ios::in | std::ios::binary); // label data
            assert(ftraind.is_open());
            assert(ftrainl.is_open());
            assert(ftestd.is_open());
            assert(ftestl.is_open());

            cuv::tensor<unsigned char,cuv::host_memory_space> traind(cuv::extents[2][1][200][200]);
            cuv::tensor<unsigned char,cuv::host_memory_space> trainl(cuv::extents[2][1][200][200]);
            cuv::tensor<unsigned char,cuv::host_memory_space> testd(cuv::extents[10][1][200][200]);
            cuv::tensor<unsigned char,cuv::host_memory_space> testl(cuv::extents[10][1][200][200]);
            ftraind.read((char*)traind.ptr(), traind.size()); assert(ftraind.good());
            ftrainl.read((char*)trainl.ptr(), trainl.size()); assert(ftrainl.good());
            ftestd.read((char*)testd.ptr(), testd.size());    assert(ftestd.good());
            ftestl.read((char*)testl.ptr(), testl.size());    assert(ftestl.good());
            //for (int i=0; i<10; i++){
                //auto img = traind[cuv::indices[i][cuv::index_range()][cuv::index_range()]];
                //img *= (unsigned char) 255;
                //cuv::libs::cimg::save(img, std::string("traind_") + boost::lexical_cast<std::string>(i) + ".png");
            //}

            train_data.resize(traind.shape());
            test_data.resize(testd.shape());
            convert(train_data , traind); // convert data type
            convert(test_data  , testd); // convert data type

            train_labels.resize(trainl.shape());
            test_labels.resize(testl.shape());
            convert(train_labels , trainl); // convert data type
            convert(test_labels  , testl); // convert data type


            binary = true;
            channels = 1;
            image_size = 200;
            std::cout << "done."<<std::endl;
        }
    };
}


#endif /* __CUVNET_LETTERS_INPAINTING_HPP__ */
