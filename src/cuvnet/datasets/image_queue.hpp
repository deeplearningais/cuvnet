#ifndef __IMAGE_QUEUE_isajfaisdf__
#define __IMAGE_QUEUE_isajfaisdf__

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <threadpool/ThreadPool.h>

namespace cuvnet{
    class model;

    struct dataset{
        virtual void load_batch(model*)=0;
    };

    struct epoch_end{};
    
    /// set of patterns which belong together, eg sliding windows over the same image
    template<class Pattern>
        struct pattern_set{
            /// the type of the patterns
            typedef Pattern pattern_type;

            /// patterns which have been generated, but still need to be
            /// processed by the network
            std::vector<boost::shared_ptr<Pattern> > m_todo;

            /// patterns which are currently being processed by the network
            std::vector<boost::weak_ptr<Pattern> > m_processing;

            /// patterns which have been processed by the network
            std::vector<boost::shared_ptr<Pattern> > m_done;

            /// add a pattern to the TODO list
            inline void push(boost::shared_ptr<pattern_type> p){
                assert(!done_generating());
                m_todo.push_back(p);
            }

            /// get a pattern for processing, also moves it from todo to processing internally.
            inline boost::shared_ptr<Pattern> get_for_processing(){
                assert(!m_todo.empty());
                boost::shared_ptr<pattern_type> pat = m_todo.back();
                m_processing.push_back(boost::weak_ptr<pattern_type>(pat));
                m_todo.pop_back();
                return pat;
            }

            struct cmp_weak_strong{
                const boost::shared_ptr<Pattern>& q;
                cmp_weak_strong(const boost::shared_ptr<Pattern>& _q):q(_q){}
                bool operator()(const boost::weak_ptr<Pattern>& p){
                    return p.lock() == q;
                }
            };

            /// tell the pattern_set that the pattern p can be moved from processing to done.
            inline void notify_processed(boost::shared_ptr<Pattern> p){
                typename std::vector<boost::weak_ptr<Pattern> >::iterator it 
                    = std::find_if(m_processing.begin(), m_processing.end(), cmp_weak_strong(p));
                assert(it != m_processing.end());
                m_processing.erase(it);
                m_done.push_back(p);
            }

            /// this is a special marker pattern which can be put in the queue to signal the end of a loop through the dataset.
            inline bool is_end_marker(){
                return m_todo.size() == 0 
                    && m_processing.size() == 0
                    && m_done.size() == 0;
            }

            inline size_t todo_size()const{return m_todo.size();}
        };

    template<class MetaDataset>
        struct image_queue{
            /// does the processing for us
            ThreadPool& m_pool;

            /// contains information about the dataset and methods to load it from disk
            MetaDataset& m_meta;

            typedef typename MetaDataset::patternset_type patternset_type;
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
                    m_queue.emplace(m_pool.enqueue(
                                [&, i, size](){
                                    return m_meta.next(i % size);
                                }));
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
