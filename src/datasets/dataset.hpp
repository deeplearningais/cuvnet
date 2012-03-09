#ifndef __CUVNET_DATASET_HPP__
#     define __CUVNET_DATASET_HPP__

#include <cuv/basics/tensor.hpp>

namespace cuvnet
{
    enum cv_mode {
        CM_TRAIN,
        CM_TRAINALL,
        CM_VALID,
        CM_TEST
    };

    class dataset
    {
        private:
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) {
                    ar & channels & image_size 
                        & train_data & val_data & test_data
                        & train_labels & val_labels & test_labels;
                }
        public:
            cuv::tensor<float,cuv::host_memory_space> train_data;
            cuv::tensor<float,cuv::host_memory_space> val_data;
            cuv::tensor<float,cuv::host_memory_space> test_data;

            cuv::tensor<int,cuv::host_memory_space> train_labels;
            cuv::tensor<int,cuv::host_memory_space> val_labels;
            cuv::tensor<int,cuv::host_memory_space> test_labels;

            int channels;
            int image_size;
            bool binary;
    };


}
#endif /* __CUVNET_DATASET_HPP__ */
