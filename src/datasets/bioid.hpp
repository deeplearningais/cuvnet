#ifndef __BIOID_HPP__
#     define __BIOID_HPP__

#include "dataset.hpp"
#include <cuv/basics/tensor.hpp>

namespace cuvnet
{

    /**
     * Access to the BioID dataset.
     * @ingroup datasets
     */
    struct bioid_dataset
        : public dataset
    {
        bioid_dataset(){
            std::cout << "Reading BioID dataset..."<<std::flush;
            const unsigned int size = 120*120;
            const unsigned int tsize = 30*30 * 2;
            train_data.resize(cuv::extents[1200][size]);
            test_data. resize(cuv::extents[321][size]);
            train_labels.resize(cuv::extents[1200][tsize]);
            test_labels.resize(cuv::extents[321][tsize]);

            {   std::ifstream ifs("/home/local/datasets/bioid/BioID_trainval_data", std::ios::binary);
                for(unsigned int j=0;j<1200*size;j++)
                    train_data[j] = (float) ifs.get();
            }
            {   std::ifstream ifs("/home/local/datasets/bioid/BioID_test_data", std::ios::binary);
                for(unsigned int j=0;j<321*size;j++)
                    test_data[j] = (float) ifs.get();
            }

            {   std::ifstream ifs("/home/local/datasets/bioid/BioID_trainval_teach", std::ios::binary);
                for(unsigned int j=0;j<1200*tsize;j++)
                    train_labels[j] = (float) ifs.get();
            }
            {   std::ifstream ifs("/home/local/datasets/bioid/BioID_test_teach", std::ios::binary);
                for(unsigned int j=0;j<321*tsize;j++)
                    test_labels[j] = (float) ifs.get();
            }
            train_labels /= 255.f;
            test_labels /= 255.f;
            train_data /= 255.f;
            test_data /= 255.f;
            binary = false;
            channels = 1;
            image_size = 120;
        }
    };

}
#endif /* __BIOID_HPP__ */
