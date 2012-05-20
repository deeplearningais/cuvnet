#ifndef __SPLITTER_HPP__
#     define __SPLITTER_HPP__
#include <cmath>
#include <vector>
#include "dataset.hpp"
#include <cuv/tensor_ops/tensor_ops.hpp>

namespace cuvnet
{
    class splitter 
    {
        private:
            dataset m_ds;     ///< the original dataset which is to be split
            unsigned int m_n_splits; ///< the number of splits
            float m_val_frac; ///< fraction of samples in the validation set
            cuv::tensor<int,cuv::host_memory_space> m_indicators; ///< values indicate number of split this belongs to
        public:
            const dataset& get_ds()const{ return m_ds; }
            unsigned int size()const{return m_n_splits;}
            splitter() : m_n_splits(0) {}
            splitter(dataset ds, unsigned int n_splits, float val_frac=0.16667)
                :m_ds(ds)
                ,m_n_splits(n_splits)
                ,m_val_frac(val_frac)
            {
                std::cout << "Splitting training_data into "<<n_splits<<" parts."<<std::endl;
            }
            void init(dataset ds, unsigned int n_splits, float val_frac=0.16667){
                m_ds = ds;
                m_n_splits = n_splits;
                m_val_frac = val_frac;
            }
            void init(dataset ds, const cuv::tensor<int,cuv::host_memory_space>&  indicators){
                m_ds = ds;
                m_n_splits = cuv::maximum(indicators)+1; // +1 for label 0
                m_val_frac = -1.f;                       // not needed
                m_indicators = indicators;
            }
			unsigned int n_splits()const{return m_n_splits; }
            dataset get_split_indicators(int idx){
                using namespace cuv;
                unsigned int n_test = cuv::count(m_indicators, idx);
                unsigned int n_left = m_indicators.size() - n_test;
                assert(n_test > 0);
                assert(n_left > 0);
                
                dataset dst;
                dst.binary     = m_ds.binary;
                dst.channels   = m_ds.channels;
                dst.image_size = m_ds.image_size;
                dst.train_data.resize(extents[n_left][m_ds.train_data.shape(1)]);
                dst.val_data.resize(extents[n_test][m_ds.train_data.shape(1)]);
                if(m_ds.train_labels.ndim()==1) {
                    dst.train_labels.resize(extents[n_left]);
                    dst.val_labels.resize(extents[n_test]);
                }else{
                    dst.train_labels.resize(extents[n_left][m_ds.train_labels.shape(1)]);
                    dst.val_labels.resize(extents[n_test][m_ds.train_labels.shape(1)]);
                }
                dst.test_data  = m_ds.test_data;
                dst.test_labels= m_ds.test_labels;

                // now copy according to indicator
                unsigned int train_idx=0, val_idx=0;
                for (unsigned int i = 0; i < m_indicators.size(); ++i)
                {
                    if(m_indicators[i]==idx){
                        // this goes in val-set
                        dst.val_data[indices[val_idx][index_range()]] = m_ds.train_data[indices[i][index_range()]];

                        if(m_ds.train_labels.ndim()==2)
                            dst.val_labels[indices[val_idx][index_range()]] = m_ds.train_labels[indices[i][index_range()]];
                        else
                            dst.val_labels(val_idx) = m_ds.train_labels(i);
                        val_idx ++;
                    }
                    else{
                        // this goes in val-set
                        dst.train_data[indices[train_idx][index_range()]] = m_ds.train_data[indices[i][index_range()]];

                        if(m_ds.train_labels.ndim()==2)
                            dst.train_labels[indices[train_idx][index_range()]] = m_ds.train_labels[indices[i][index_range()]];
                        else
                            dst.train_labels(train_idx) = m_ds.train_labels(i);
                        train_idx ++;
                    }
                }

                return dst;
            }
            dataset operator[](unsigned int idx){ 
                using namespace cuv;
                cuvAssert(idx<m_n_splits);
                if(m_indicators.ptr()){
                    return get_split_indicators(idx);
                }
                unsigned int n_examples = m_ds.train_data.shape(0);
                unsigned int start,end;
                if(m_n_splits==1){
                    // just split in training and validation using m_val_frac
                    // the last m_val_frac*n_examples items are used for validation.
                    start = n_examples * (1.f-m_val_frac);
                    end   = n_examples;
                }else{
                    unsigned int step       = std::ceil(n_examples/(float)m_n_splits);
                    start  = step*idx;
                    end    = std::min(n_examples,start+step);
                }
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
