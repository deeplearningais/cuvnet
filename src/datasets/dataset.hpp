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

    /**
     * Generic 'simple' dataset which completely fits in RAM.
     *
     * @ingroup datasets
     */
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
            cuv::tensor<float,cuv::host_memory_space> train_data; ///< contains the training data
            cuv::tensor<float,cuv::host_memory_space> val_data; ///< contains the validation data
            cuv::tensor<float,cuv::host_memory_space> test_data; ///< contains the test data

            cuv::tensor<int,cuv::host_memory_space> train_labels; ///< contains the training labels
            cuv::tensor<int,cuv::host_memory_space> val_labels; ///< contains the validation labels
            cuv::tensor<int,cuv::host_memory_space> test_labels; ///< contains the test labels

            int channels; ///< the number of channels, if the dataset contains images
            int image_size; ///< the size of the images (width==height) if dataset contains images
            bool binary; ///< if true, assume that the dataset contains Bernoulli-distributed data
    };


}
#endif /* __CUVNET_DATASET_HPP__ */
