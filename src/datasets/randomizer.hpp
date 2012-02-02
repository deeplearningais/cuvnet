#ifndef __RANDOMIZER_HPP__
#     define __RANDOMIZER_HPP__
#include <vector>
#include "dataset.hpp"

namespace cuvnet
{
    class randomizer 
    {
        public:
            randomizer() { }
            dataset transform(dataset& ds)const{
                using namespace cuv;
                dataset dst;
                dst.test_data = ds.test_data;
                dst.test_labels = ds.test_labels;

                dst.channels   = ds.channels;
                dst.image_size = ds.image_size;

                dst.train_data   = tensor<float,host_memory_space>(ds.train_data.shape());
                dst.train_labels = tensor<int  ,host_memory_space>(ds.train_labels.shape());
                unsigned int n_train_data = ds.train_data.shape(0);
                std::vector<unsigned int> idx(n_train_data);
                for (unsigned int i = 0; i < n_train_data; ++i)
                    idx[i]=i;
                std::random_shuffle(idx.begin(),idx.end());
                for(unsigned int i=0;i<n_train_data;i++){
                    tensor_view<float,host_memory_space> srcv = ds .train_data[indices[idx[i]][index_range()]];
                    tensor_view<float,host_memory_space> dstv = dst.train_data[indices[i][index_range()]];
                    dstv = srcv;
                }
                if(ds.train_labels.ndim()==1){
                    for(unsigned int i=0;i<n_train_data;i++){
                        dst.train_labels[i] = ds.train_labels[idx[i]];
                    }
                }else if(ds.train_labels.ndim()==2){
                    for(unsigned int i=0;i<n_train_data;i++){
                        tensor_view<int,host_memory_space> srcv = ds .train_labels[indices[idx[i]][index_range()]];
                        tensor_view<int,host_memory_space> dstv = dst.train_labels[indices[i][index_range()]];
                        dstv = srcv;
                    }
                }else{
                    throw std::runtime_error("unknown data format");
                }
                return dst;
            }
    };
}
#endif /* __RANDOMIZER_HPP__ */
