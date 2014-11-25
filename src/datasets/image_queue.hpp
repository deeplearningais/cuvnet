#ifndef __IMAGE_QUEUE_HPP__
#     define __IMAGE_QUEUE_HPP__

#include <queue>
#include <boost/thread.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <datasets/bounding_box_tools.hpp>

namespace cuvnet
{

    namespace image_datasets
    {
        struct base_image_queue{
            virtual void notify_patternset_complete()=0;
        };

        /// a fully loaded pattern.                                                                                                                                     
        struct classification_pattern{                                                                                                                                                 
            bbtools::image_meta_info meta_info;                                                                                                                         
            cuv::tensor<float,cuv::host_memory_space> img; ///< image                                                                                                   
            cuv::tensor<float,cuv::host_memory_space> tch; ///< teacher
            cuv::tensor<float,cuv::host_memory_space> result; ///< result                                                                                               
        };       

        /// set of patterns which belong together, eg sliding windows over the same image
        template<class Pattern>
        struct pattern_set{
            /// the type of the patterns
            typedef Pattern pattern_type;
            
            /// this gets notified when all patterns have been generated so that waiting threads can be woken up.
            base_image_queue* m_queue;

            /// this is set to true once m_todo has been filled completely
            bool m_done_generating; 

            /// patterns which have been generated, but still need to be
            /// processed by the network
            std::vector<boost::shared_ptr<Pattern> > m_todo;

            /// patterns which are currently being processed by the network
            std::vector<boost::weak_ptr<Pattern> > m_processing;

            /// patterns which have been processed by the network
            std::vector<boost::shared_ptr<Pattern> > m_done;

            /// ctor.
            pattern_set(base_image_queue* q):m_queue(q), m_done_generating(false){}

            /// add a pattern to the TODO list
            inline void push(boost::shared_ptr<pattern_type> p, bool last=false){
                assert(!done_generating());
                m_todo.push_back(p);
                if(last)
                    set_done_generating();
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
                return m_done_generating == true 
                    && m_todo.size() == 0 
                    && m_processing.size() == 0
                    && m_done.size() == 0;
            }

            inline void set_done_generating(){ m_done_generating = true; m_queue->notify_patternset_complete(); }
            inline bool done_generating()const{ return m_done_generating; }
            inline size_t todo_size()const{return m_todo.size();}
        };


        /** a convenience wrapper around std::queue, which provides a mutex and
         *  a bulk-extraction method.
         *
         * Many image_loader instances can put their results in this queue. 
         * The patterns can then be processed in a batch fashion by popping
         * many at a time.
         */
        template<class PatternSetType>
        class image_queue : public base_image_queue {
            public:
                typedef image_queue<PatternSetType> my_type;
                typedef typename PatternSetType::pattern_type pattern_type;
                typedef PatternSetType patternset_type;
            private:
                mutable boost::mutex m_mutex;
                boost::condition_variable m_cond;
                std::queue<boost::shared_ptr<PatternSetType> > m_queue;
                bool m_signal_restart;
                log4cxx::LoggerPtr m_log;
            public:
                image_queue(bool signal_restart)
                :m_signal_restart(signal_restart){
                    m_log = log4cxx::Logger::getLogger("image_queue");
                    LOG4CXX_INFO(m_log, "Creating image queue. signal_restart: " << signal_restart);
                }

                /**
                 * New patternsets should be pushed by the thread creating the jobs.
                 * Once all the tasks inside a patternset are generated, the
                 * image loader should call notify_patternset_complete(), so
                 * that waiting threads are woken up.
                 */
                void push(boost::shared_ptr<PatternSetType> pat, bool lock=true){ 
                    if(!lock){
                        m_queue.push(pat); 
                        m_cond.notify_one();
                        return;
                    }
                    {
                        boost::mutex::scoped_lock l(m_mutex);
                        m_queue.push(pat); 
                    }
                }

                virtual void notify_patternset_complete() override {
                    m_cond.notify_one();
                }

                /// return the number of patterns currently stored in the queue
                size_t size()const{
                    return m_queue.size();
                }
                /// return whether size is >= value
                bool can_pop()const{
                    return m_queue.size() >= 1
                        && m_queue.front()->done_generating();
                }

                /// remove all patterns from the queue
                void clear(){
                    m_queue.clear();
                }

                /**
                 * Get a pattern from the queue.
                 * Blocks when queue is empty.
                 *
                 * @param dest where to put the patterns
                 */
                boost::shared_ptr<pattern_type> 
                pop(){
                    boost::mutex::scoped_lock lock(m_mutex);
                    m_cond.wait(lock, boost::bind(&my_type::can_pop, this));

                    while(m_queue.front()->is_end_marker()){
                        LOG4CXX_DEBUG(m_log, "encountered end marker");
                        m_queue.pop();
                        if(m_signal_restart)
                            throw epoch_end();
                        else
                            m_cond.wait(lock, boost::bind(&my_type::can_pop, this));
                    }

                    // retrieve object to return
                    boost::shared_ptr<pattern_type> ret 
                        = m_queue.front()->get_for_processing();

                    // check whether the set is empty now
                    if(m_queue.front()->todo_size() == 0){
                        m_queue.pop();
                    }

                    return ret;
                }
        };

        /**
         * Provides access to a VOC-type dataset with annotations per image.
         *
         * The access is either shuffled or non-shuffled.
         */
        class image_dataset{
            private:
                std::vector<bbtools::image_meta_info> m_dataset;
                std::vector<unsigned int> m_indices;
                bool m_shuffle;
                void read_meta_info(std::vector<bbtools::image_meta_info>& dest, const std::string& filename);
                log4cxx::LoggerPtr m_log;
            public:
                /**
                 * ctor.
                 * @param filename file storing infos of the dataset
                 * @param shuffle if true, load images in random order
                 */
                image_dataset(const std::string& filename, bool shuffle);
                
                /**
                 * get meta-infos for i-th image in the dataset.
                 */
                const bbtools::image_meta_info& get(unsigned int i)const{
                    return m_dataset[m_indices[i]];
                }

                /**
                 * shuffle indices.
                 */
                inline void shuffle(){
                    if(m_shuffle){
                        LOG4CXX_INFO(m_log, "Shuffling.");
                        std::random_shuffle(m_indices.begin(), m_indices.end());
                    }
                }
               
                /**
                 * return number of images in the dataset.
                 */
                size_t size()const{
                    return m_indices.size();
                }
        };

        /** 
         * Loads one image into a single pattern. 
         */
        class sample_image_loader{
            private:
                typedef pattern_set<classification_pattern> patternset_type;
                typedef boost::shared_ptr<patternset_type> patternset_ptr;
                patternset_ptr m_dst;
                const bbtools::image_meta_info* m_meta;      ///< meta-infos for the image to be processed

                bool m_grayscale;
                unsigned int m_pattern_size;
                unsigned int m_n_classes;
                log4cxx::LoggerPtr m_log;
            public:
                /**
                 * @param queue where to put the results
                 * @param meta the image to load
                 * @param pattern_size width/height of patterns
                 * @param grayscale if true, discard color information
                 * @param n_classes the number of classes that can be distinguished (=number of output maps).
                 */
                sample_image_loader(patternset_ptr dst,
                        const bbtools::image_meta_info* meta, 
                        unsigned int pattern_size, 
                        bool grayscale,
                        unsigned int n_classes);

                virtual void operator()();
        };

        namespace detail
        {
            struct asio_queue_impl;
            struct asio_queue{
                boost::shared_ptr<asio_queue_impl> m_impl;
                unsigned int m_n_threads;
                asio_queue(unsigned int n_threads);
                inline unsigned int n_threads(){ return m_n_threads; }
                void stop();
                void post(boost::function<void (void)>);
            };
        }
        /**
         * starts many workers which process images to patterns and enqueue
         * them into an image queue.
         *
         *                   _-----> worker ---_
         *                  /                   \
         * ---> dataset --> +------> worker -----+---> image queue ---> ...
         *                  \_                  /
         *                    -----> worker ---^
         *
         * Running in a separate thread, it ensures that the image queue always
         * has a certain "fill level".
         *
         * @tparam Queue the type of the queue where patterns are stored
         *
         * @tparam Dataset the dataset type, where images are taken from.
         *         Must support a size() operation that returns the number of
         *         images in the dataset and a get(int) function that returns a
         *         bbtools::image_meta_info.
         *
         * @tparam WorkerFactory should return an image-loader, that, given the
         *         queue and the image-meta-information,
         *
         *         - loads the image
         *         - pre-processes the image if needed
         *         - pushs (whole or parts of the image) in the queue
         */
        template<class Queue, class Dataset, class WorkerFactory>
        class loader_pool{
            private:
                typedef loader_pool<Queue,Dataset,WorkerFactory> my_type;
                typedef typename Queue::patternset_type patternset_type;

                /// this keeps the results of the worker processing, which can
                /// be then processed by the network
                Queue* m_result_queue;

                /// contains references (such as filenames) to the data vectors
                /// which are stored in some mass memory.
                Dataset* m_dataset;

                /// contains all the worker threads
                boost::thread m_pool_thread;

                /// if true, the workers have just received jobs and we
                /// shouldn't create new jobs until this variable was set to
                /// false and the result queue size is smaller than
                /// m_min_pipe_len.
                bool m_running;
                bool m_request_stop;

                /// Usually a function that creates worker jobs.
                /// If your workers need parameters, this is the way to pass
                /// them. The worker factory gets passed two parameters,
                /// the pattern_set to work on and a pointer to the dataset
                /// element to be processed.
                WorkerFactory m_worker_factory;

                log4cxx::LoggerPtr m_log;

                /// The queue should never have less than m_min_pipe_len and
                /// never more than  m_max_pipe_len elements. Due to parallel
                /// processing, this cannot be guaranteed exactly.
                unsigned int m_min_pipe_len, m_max_pipe_len;

                /// this keeps the jobs to be processed by the worker threads
                detail::asio_queue m_asio_queue;
                void on_stop(){
                    m_running = false;
                }
            public:

                /**
                 * ctor.
                 * @param queue where to put patterns once they're ready
                 * @param n_threads how many threads to use, 0 means number of CPUs
                 * @param ds the dataset to load images from
                 * @param worker_factory generates a worker that takes meta-information and puts patterns in the queue.
                 * @param min_queue_len minimal length of the queue
                 * @param max_queue_len maximal length of the queue
                 */
                loader_pool(Queue& queue, unsigned int n_threads, Dataset& ds, WorkerFactory worker_factory, size_t min_queue_len=32, size_t max_queue_len=32*3)
                    : m_result_queue(&queue)
                    , m_dataset(&ds)
                    , m_running(false)
                    , m_request_stop(false)
                    , m_worker_factory(worker_factory)
                    , m_min_pipe_len(min_queue_len)
                    , m_max_pipe_len(max_queue_len)
                    , m_asio_queue(n_threads)
                {
                    m_log = log4cxx::Logger::getLogger("loader_pool");
                }
                /**
                 * start filling the queue asynchronously.
                 */
                void start(){
                    m_pool_thread = boost::thread(boost::ref(*this));
                    LOG4CXX_INFO(m_log, "started");
                }

                /**
                 * stop the asynchronous queue-filling.
                 */
                void request_stop(){
                    m_request_stop = true;
                    LOG4CXX_INFO(m_log, "stop requested");
                    m_pool_thread.join();
                    LOG4CXX_INFO(m_log, "finished");
                }

                /**
                 * join all threads
                 */
                void stop(){
                    request_stop();
                    m_asio_queue.stop();
                }

                /// dtor.
                ~loader_pool(){
                    request_stop();
                }

                /** 
                 * main loop, ensures queue stays filled and stops when \c m_request_stop is true.
                 */
                void operator()(){
                    unsigned int cnt = 0;
                    m_running = false;
                    while(!m_request_stop){
                        unsigned int size = m_result_queue->size();
                        if(size < m_min_pipe_len && !m_running){
                            for (unsigned int i = 0; i < m_min_pipe_len && i + size < m_max_pipe_len; ++i)
                            {
                                boost::shared_ptr<patternset_type> todo
                                    = boost::make_shared<patternset_type>(m_result_queue);
                                m_result_queue->push(todo);

                                m_asio_queue.post(m_worker_factory(todo, &m_dataset->get(cnt)));
                                cnt = (cnt+1) % m_dataset->size();
                                if(cnt == 0){
                                    LOG4CXX_INFO(m_log, "Roundtrip through dataset (" << m_dataset->size()<< ") completed.");
                                    // additionally queue an "end" marker
                                    boost::shared_ptr<patternset_type> end_marker
                                        = boost::make_shared<patternset_type>(m_result_queue);
                                    end_marker->set_done_generating();
                                    m_result_queue->push(end_marker);
                                    m_dataset->shuffle();
                                }
                            }

                            // the last job should set m_running to false;
                            m_running = true;
                            m_asio_queue.post(boost::bind(&my_type::on_stop, this));
                        }
                        boost::this_thread::sleep(boost::posix_time::millisec(10));
                    }
                    on_stop();
                }
        };

        /**
         * convenience function for creating loader_pool objects.
         * @see loader_pool
         */
        template<class Queue, class Dataset, class WorkerFactory>
        boost::shared_ptr<loader_pool<Queue, Dataset, WorkerFactory> >
        make_loader_pool(unsigned int n_threads, Queue& q, Dataset& ds, WorkerFactory wf, int min_queue_len=32, int max_queue_len=32*3){
            typedef loader_pool<Queue, Dataset, WorkerFactory> lp_t;
            typedef boost::shared_ptr<lp_t> ret_t;
            return ret_t(new lp_t(q, n_threads, ds, wf, min_queue_len, max_queue_len));
        }

    }
}


#endif /* __IMAGE_QUEUE_HPP__ */
