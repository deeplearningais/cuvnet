#ifndef __AMAT_DATASETS_HPP__
#     define __AMAT_DATASETS_HPP__

#include <tools/read_amat.hpp>
#include "dataset.hpp"

namespace cuvnet
{
    struct amat_dataset
        : public dataset
    {
        amat_dataset(const std::string& zipfile, const std::string& train, const std::string& test){
            read_amat_with_label(train_data,train_labels,zipfile,train);
            read_amat_with_label(test_data,test_labels,zipfile,test);
            channels = 1;
            image_size = 28;
        }
    };

}
#endif /* __AMAT_DATASETS_HPP__ */
