#ifndef __IMAGE_QUEUE_isajfaisdf__
#define __IMAGE_QUEUE_isajfaisdf__

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <threadpool/ThreadPool.h>

namespace cuvnet{
    /**
     * Exception thrown by datasets when the 
     */
    struct epoch_end{};

    /**
     * Asynchronous loading of meta-datasets.
     * 
     * A meta-dataset must have: a size() function, returning the number of elements in the data set and
     * a next(size_t idx) function, returning element with index idx as a PatternSet.
     *
     * Every member of a patternset is an element which can be processed by a model.
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
                :m_pool(pool), m_meta(meta_dataset), m_min_size(min_size), m_max_size(max_size), m_idx(0){}

            /// schedule new jobs (run if queue size is below m_min_size when pop is called)
            void schedule_new_jobs(int n_jobs){
                size_t size = m_meta.size();

                for(size_t i=m_idx; i < m_idx + n_jobs; i++){
                    if(m_pool.size() > 0)
                        m_queue.emplace(m_pool.enqueue(
                                    [&, i, size](){
                                    return m_meta.next(i % size);
                                    }));
                    else{
                        // synchronous processing
                        std::promise<boost::shared_ptr<patternset_type> > p;
                        p.set_value(m_meta.next(i % size));
                        m_queue.emplace(p.get_future());
                    }
                }
                m_idx = (m_idx + n_jobs) % size;
            }

            /// return one job for the model
            boost::shared_ptr<pattern_type> pop(){
                if(m_queue.size() < m_min_size)
                    schedule_new_jobs(m_max_size - (int)m_queue.size());

                // queue is now non-empty
                boost::shared_ptr<patternset_type> pat
                    = m_queue.front().get();

                if(pat->is_end_marker()){
                    m_queue.pop();
                    throw epoch_end();
                }
                
                boost::shared_ptr<pattern_type> ret = 
                    pat->get_for_processing();
                if(pat->todo_size() == 0)
                    m_queue.pop();
                return ret;
            }

            private:
                size_t m_idx;
        };
};

#endif /* __IMAGE_QUEUE_isajfaisdf__ */
