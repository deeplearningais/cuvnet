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
            void transform(
                    cuv::tensor<float,cuv::host_memory_space>& train_data, 
                    cuv::tensor<int,cuv::host_memory_space>& train_labels, 
                    cuv::tensor<int,cuv::host_memory_space>* p_bag_index=NULL)const{

                using namespace cuv;
                tensor<float,host_memory_space> new_train_data   (train_data.shape());
                tensor<int  ,host_memory_space> new_train_labels (train_labels.shape());
                tensor<int  ,host_memory_space> new_bag_index ;

                unsigned int n_train_data = train_data.shape(0);

                std::vector<unsigned int> idx(n_train_data);
                for (unsigned int i = 0; i < n_train_data; ++i)
                    idx[i]=i;
                std::random_shuffle(idx.begin(),idx.end());

                for(unsigned int i=0;i<n_train_data;i++){
                    tensor_view<float,host_memory_space> srcv = train_data[indices[idx[i]][index_range()]];
                    tensor_view<float,host_memory_space> dstv = new_train_data[indices[i][index_range()]];
                    dstv = srcv;
                }
                train_data = new_train_data;

                if(p_bag_index != NULL){
                    cuv::tensor<int,cuv::host_memory_space>& bag_index = *p_bag_index;
                    cuvAssert(bag_index.size() == n_train_data);
                    new_bag_index.resize(bag_index.shape());
                    for(unsigned int i=0;i<n_train_data;i++){
                        new_bag_index[i] = bag_index[idx[i]];
                    }
                    bag_index = new_bag_index;
                }

                if(train_labels.ndim()==1){
                    for(unsigned int i=0;i<n_train_data;i++){
                        new_train_labels[i] = train_labels[idx[i]];
                    }
                }else if(train_labels.ndim()==2){
                    for(unsigned int i=0;i<n_train_data;i++){
                        tensor_view<int,host_memory_space> srcv = train_labels[indices[idx[i]][index_range()]];
                        tensor_view<int,host_memory_space> dstv = new_train_labels[indices[i][index_range()]];
                        dstv = srcv;
                    }
                }else{
                    throw std::runtime_error("unknown data format");
                }
                train_labels = new_train_labels;
            }
    };
}
#endif /* __RANDOMIZER_HPP__ */
