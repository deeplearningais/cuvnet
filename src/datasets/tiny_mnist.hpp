#include "dataset.hpp"
struct tiny_mnist : public cuvnet::dataset{
    tiny_mnist(const std::string path){
        train_data.resize(cuv::extents[60000][30]);
        test_data.resize(cuv::extents[10000][30]);
        train_labels.resize(cuv::extents[60000][10]);
        test_labels.resize(cuv::extents[10000][10]);

        {
            std::ifstream ifs((path + "/train-30-60000-float32.dat").c_str());
            assert(ifs.is_open());
            ifs.read((char*)train_data.ptr(), sizeof(float)*train_data.size());
        }
        {
            std::ifstream ifs((path + "/test-30-10000-float32.dat").c_str());
            assert(ifs.is_open());
            ifs.read((char*)test_data.ptr(), sizeof(float)*test_data.size());
        }
        {
            std::ifstream ifs((path + "/train-labels-10-60000-float32.dat").c_str());
            assert(ifs.is_open());
            ifs.read((char*)train_labels.ptr(), sizeof(float)*train_labels.size());
        }
        {
            std::ifstream ifs((path + "/test-labels-10-10000-float32.dat").c_str());
            assert(ifs.is_open());
            ifs.read((char*)test_labels.ptr(), sizeof(float)*test_labels.size());
        }

        std::cout << "train_data: "<< cuv::minimum(train_data) << " <= "<<cuv::maximum(train_data) << std::endl;
        binary = false;
    }
};
