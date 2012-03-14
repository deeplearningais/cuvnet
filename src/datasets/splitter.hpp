#ifndef __SPLITTER_HPP__
#     define __SPLITTER_HPP__
#include <cmath>
#include <vector>
#include "dataset.hpp"

namespace cuvnet
{
    class splitter 
    {
        private:
            dataset m_ds;
            unsigned int m_n_splits;
        public:
            const dataset& get_ds()const{ return m_ds; }
            unsigned int size()const{return m_n_splits;}
            splitter() : m_n_splits(0) {}
            splitter(dataset ds, unsigned int n_splits)
                :m_ds(ds)
                ,m_n_splits(n_splits)
            {
                std::cout << "Splitting training_data into "<<n_splits<<" parts."<<std::endl;
            }
            void init(dataset ds, unsigned int n_splits){
                m_ds = ds;
                m_n_splits = n_splits;
            }
			unsigned int n_splits()const{return m_n_splits; }
            dataset operator[](unsigned int idx){ 
                using namespace cuv;
                cuvAssert(idx<m_n_splits);
                unsigned int n_examples = m_ds.train_data.shape(0);
                unsigned int step       = std::ceil(n_examples/(float)m_n_splits);
                unsigned int start      = step*idx;
                unsigned int end        = std::min(n_examples,start+step);
                unsigned int n_left     = n_examples - (end-start);
                dataset dst;
                dst.binary     = m_ds.binary;
                dst.channels   = m_ds.channels;
                dst.image_size = m_ds.image_size;
                dst.train_data.resize(extents[n_left][m_ds.train_data.shape(1)]);
                dst.val_data   = m_ds.train_data  [indices[index_range(start,end)][index_range()]];
                dst.test_data  = m_ds.test_data;
                dst.test_labels= m_ds.test_labels;
                if(m_ds.train_labels.ndim()==1) {
                    dst.train_labels.resize(extents[n_left]);
                    dst.val_labels   = m_ds.train_labels[indices[index_range(start,end)]];
                }
                else                           {
                    dst.train_labels.resize(extents[n_left][m_ds.train_labels.shape(1)]);
                    dst.val_labels   = m_ds.train_labels[indices[index_range(start,end)][index_range()]];
                }
                if(start>0){
                    {
                        tensor_view<float,host_memory_space> before_src(indices[index_range(0,start)][index_range()], m_ds.train_data);
                        tensor_view<float,host_memory_space> before_dst(indices[index_range(0,start)][index_range()], dst.train_data);
                        before_dst = before_src;
                    }
                    if(m_ds.train_labels.ndim()==1){
                        tensor_view<int,host_memory_space> before_src(indices[index_range(0,start)], m_ds.train_labels);
                        tensor_view<int,host_memory_space> before_dst(indices[index_range(0,start)], dst.train_labels);
                        before_dst = before_src;
                    } else{
                        tensor_view<int,host_memory_space> before_src(indices[index_range(0,start)][index_range()], m_ds.train_labels);
                        tensor_view<int,host_memory_space> before_dst(indices[index_range(0,start)][index_range()], dst.train_labels);
                        before_dst = before_src;
                    }
                }
                if(end<n_examples){
                    {
                        tensor_view<float,host_memory_space> after_src(indices[index_range(end,n_examples)][index_range()], m_ds.train_data);
                        tensor_view<float,host_memory_space> after_dst(indices[index_range(start,n_left)][index_range()], dst.train_data);
                        after_dst = after_src;
                    }
                    if(m_ds.train_labels.ndim()==1){
                        tensor_view<int,host_memory_space> after_src(indices[index_range(end,n_examples)], m_ds.train_labels);
                        tensor_view<int,host_memory_space> after_dst(indices[index_range(start,n_left)], dst.train_labels);
                        after_dst = after_src;
                    } else{
                        tensor_view<int,host_memory_space> after_src(indices[index_range(end,n_examples)][index_range()], m_ds.train_labels);
                        tensor_view<int,host_memory_space> after_dst(indices[index_range(start,n_left)][index_range()], dst.train_labels);
                        after_dst = after_src;
                    }
                }
                return dst;
            }
    };
}
#endif /* __SPLITTER_HPP__ */
