#ifndef __SPLITTER_HPP__
#     define __SPLITTER_HPP__
#include <vector>
#include "dataset.hpp"

namespace cuvnet
{
    class splitter 
    {
        private:
            std::vector<dataset> datasets;
        public:
            dataset& operator[](unsigned int i){ return datasets[i]; }
            unsigned int size()const{return datasets.size();}
            splitter(dataset& ds, unsigned int n_splits){
                std::cout << "Splitting dataset into "<<n_splits<<" parts..."<<std::flush;
                using namespace cuv;
                unsigned int n_examples = ds.train_data.shape(0);
                unsigned int step = ceil(n_examples/(float)n_splits);
                for(unsigned int i=0;i<n_examples;i+=step){
                    unsigned int start = i;
                    unsigned int end   = std::min(n_examples,start+step);
                    unsigned int n_left = n_examples - (end-start);
                    datasets.push_back(dataset());
                    dataset& dst = datasets.back();
                    dst.channels = ds.channels;
                    dst.image_size = ds.image_size;
                    dst.val_data = ds.train_data[indices[index_range(start,end)][index_range()]];
                    dst.train_data.resize(extents[n_left][ds.train_data.shape(1)]);
                    if(dst.train_labels.ndim()==1) dst.train_labels.resize(extents[n_left]);
                    else                           dst.train_labels.resize(extents[n_left][ds.train_labels.shape(1)]);
                    if(start>0){
                        {
                            tensor_view<float,host_memory_space> before_src(indices[index_range(0,start)][index_range()], ds.train_data);
                            tensor_view<float,host_memory_space> before_dst(indices[index_range(0,start)][index_range()], dst.train_data);
                            before_dst = before_src;
                        }
                        if(ds.train_labels.ndim()==1){
                            tensor_view<int,host_memory_space> before_src(indices[index_range(0,start)], ds.train_labels);
                            tensor_view<int,host_memory_space> before_dst(indices[index_range(0,start)], dst.train_labels);
                            before_dst = before_src;
                        } else{
                            tensor_view<int,host_memory_space> before_src(indices[index_range(0,start)][index_range()], ds.train_labels);
                            tensor_view<int,host_memory_space> before_dst(indices[index_range(0,start)][index_range()], dst.train_labels);
                            before_dst = before_src;
                        }
                    }
                    if(end<n_examples){
                        {
                            tensor_view<float,host_memory_space> after_src(indices[index_range(end,n_examples)][index_range()], ds.train_data);
                            tensor_view<float,host_memory_space> after_dst(indices[index_range(start,n_left)][index_range()], dst.train_data);
                            after_dst = after_src;
                        }
                        if(ds.train_labels.ndim()==1){
                            tensor_view<int,host_memory_space> before_src(indices[index_range(end,n_examples)], ds.train_labels);
                            tensor_view<int,host_memory_space> before_dst(indices[index_range(start,n_left)], dst.train_labels);
                            before_dst = before_src;
                        } else{
                            tensor_view<int,host_memory_space> before_src(indices[index_range(end,n_examples)][index_range()], ds.train_labels);
                            tensor_view<int,host_memory_space> before_dst(indices[index_range(start,n_left)][index_range()], dst.train_labels);
                            before_dst = before_src;
                        }
                    }
                    std::cout <<"."<<std::flush;
                }
                std::cout << std::endl;
            }
    };
}
#endif /* __SPLITTER_HPP__ */
