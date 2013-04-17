#include "cuv.hpp"
#include "npy_datasets.hpp"
#include <third_party/cnpy/cnpy.h>


namespace cuvnet
{
    npy_dataset::npy_dataset(const std::string& path){
        {
            cnpy::NpyArray npytrain = cnpy::npy_load(path + "/train.npy");
            cnpy::NpyArray npytest = cnpy::npy_load(path + "/test.npy");
            train_data.resize(npytrain.shape);
            test_data.resize(npytest.shape);
            memcpy(train_data.ptr(), npytrain.data, train_data.size() * sizeof(float));
            memcpy(test_data.ptr(), npytest.data, test_data.size() * sizeof(float));
        }

        {
            cnpy::NpyArray npytrain = cnpy::npy_load(path + "/train_labels.npy");
            cnpy::NpyArray npytest = cnpy::npy_load(path + "/test_labels.npy");

            unsigned int n_classes = 0;
            for(unsigned int i=0; i < npytrain.shape[0]; i++)
                n_classes = std::max((unsigned int)npytrain.data[i], n_classes);

            train_labels.resize(cuv::extents[npytrain.shape[0]][n_classes+1]);
            test_labels.resize(cuv::extents[npytest.shape[0]][n_classes+1]);
            train_labels = 0.f;
            test_labels = 0.f;

            for(unsigned int i=0; i < npytrain.shape[0]; i++)
                train_labels(i, npytrain.data[i]) = 1.f;

            for(unsigned int i=0; i < npytest.shape[0]; i++)
                test_labels(i, npytest.data[i]) = 1.f;
        }
    }
}
