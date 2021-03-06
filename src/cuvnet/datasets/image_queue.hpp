#ifndef __IMAGE_QUEUE_isajfaisdf__
#define __IMAGE_QUEUE_isajfaisdf__

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <third_party/threadpool/ThreadPool.h>

namespace cuvnet{
    /**
     * Exception thrown by image queue when the epoch is over
     */
    struct epoch_end;

    /**
     * Asynchronous loading of meta-datasets.
     * 
     * A meta-dataset must have: a size() function, returning the number of elements in the data set and
     * a next(size_t idx) function, returning element with index idx as a PatternSet.
     *
     * Every member of a patternset is an element which can be processed by a model.
     * 
     * @ingroup datasets
     */
    template<class MetaDataset>
        struct image_queue{
            /// does the processing for us
            ThreadPool& m_pool;

            /// contains information about the dataset and methods to load it from disk
            MetaDataset& m_meta;

            typedef typename decltype(m_meta.next(0))::element_type patternset_type;
            typedef typename patternset_type::pattern_type pattern_type;

            /// schedule new jobs if queue length is below this value
            int m_min_size;
            
            /// when scheduling new jobs, total queue size should not exceed this value
            int m_max_size;

            /// contains the futures to patterns we've enqueued in the pool
            std::queue<std::future<boost::shared_ptr<patternset_type> > > m_queue;

            /// constructor
            image_queue(ThreadPool& pool, MetaDataset& meta_dataset, int min_size, int max_size)
                :m_pool(pool), m_meta(meta_dataset), m_min_size(min_size), m_max_size(max_size), m_idx(0), m_signal_end(false){}

            /// throw epoch_end exception when epoch end is encountered (defaults to false)
            inline void set_signal_epoch_end(bool b=true){ m_signal_end = b; }

            /// schedule new jobs (run if queue size is below m_min_size when pop is called)
            void schedule_new_jobs(int n_jobs){
                size_t size = m_meta.size();

                for(size_t i=m_idx; i < m_idx + n_jobs; i++){
                    if(i == size && m_signal_end){
                        std::promise<boost::shared_ptr<patternset_type> > p;
                        p.set_value(boost::make_shared<patternset_type>());
                        m_queue.emplace(p.get_future());
                    }
                    size_t idx = m_meta.shuffled_idx(i % size);
                    if(m_pool.size() > 0)
                        m_queue.emplace(m_pool.enqueue(
                                    [&, idx, size](){
                                    return m_meta.next(idx);
                                    }));
                    else{
                        // synchronous processing
                        std::promise<boost::shared_ptr<patternset_type> > p;
                        p.set_value(m_meta.next(idx));
                        m_queue.emplace(p.get_future());
                    }
                }
                if(m_idx + n_jobs > size){
                    m_meta.shuffle();
                }
                m_idx = (m_idx + n_jobs) % size;
            }

            boost::shared_ptr<patternset_type> m_current_set;

            /// return one job for the model
            boost::shared_ptr<pattern_type> pop(){
                if(m_queue.size() < m_min_size)
                    schedule_new_jobs(m_max_size - (int)m_queue.size());
                // queue is now non-empty

                if(m_current_set && m_current_set->todo_size() > 0){
                    return m_current_set->get_for_processing();
                }

                // nothing in current set
                m_current_set = m_queue.front().get();
                m_queue.pop();

                if(m_current_set->is_end_marker()){
                    m_queue.pop();
                    throw epoch_end();
                }
                
                return m_current_set->get_for_processing();
            }

            private:
                size_t m_idx;

                /// if false, don't throw an epoch_end exception when starting a new epoch
                bool m_signal_end;
        };
};

#endif /* __IMAGE_QUEUE_isajfaisdf__ */
