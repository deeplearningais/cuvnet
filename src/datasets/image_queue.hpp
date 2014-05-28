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
        /// a fully loaded pattern.                                                                                                                                     
        struct classification_pattern{                                                                                                                                                 
            bbtools::image_meta_info meta_info;                                                                                                                         
            cuv::tensor<float,cuv::host_memory_space> img; ///< image                                                                                                   
            cuv::tensor<float,cuv::host_memory_space> tch; ///< teacher
            cuv::tensor<float,cuv::host_memory_space> result; ///< result                                                                                               
        };       


        /** a convenience wrapper around std::queue, which provides a mutex and
         *  a bulk-extraction method.
         *
         * Many image_loader instances can put their results in this queue. 
         * The patterns can then be processed in a batch fashion by popping
         * many at a time.
         */
        template<class PatternType>
        class image_queue{
            private:
                typedef image_queue<PatternType> my_type;
                mutable boost::mutex m_mutex;
                boost::condition_variable m_cond;
                int m_patterns_to_epoch_end;
                std::queue<boost::shared_ptr<PatternType> > m_queue;
                bool m_signal_restart;
                log4cxx::LoggerPtr m_log;
            public:
                image_queue(bool signal_restart)
                :m_signal_restart(signal_restart){
                    m_log = log4cxx::Logger::getLogger("image_queue");
                    LOG4CXX_INFO(m_log, "Creating image queue. signal_restart: " << signal_restart);
                    m_patterns_to_epoch_end = -1;
                }

                void on_epoch_ends(){
                    m_patterns_to_epoch_end = size();
                }

                void push(boost::shared_ptr<PatternType> pat, bool lock=true){ 
                    if(!lock){
                        m_queue.push(pat); 
                        m_cond.notify_one();
                        return;
                    }
                    {
                        boost::mutex::scoped_lock l(m_mutex);
                        m_queue.push(pat); 
                    }
                    m_cond.notify_one();
                }

                /// return the number of patterns currently stored in the queue
                size_t size()const{
                    return m_queue.size();
                }
                /// return whether size is >= value
                bool size_ge_val(size_t value)const{
                    return m_queue.size() >= value;
                }

                /// remove all patterns from the queue
                void clear(){
                    m_queue.clear();
                }

                /**
                 * Get a number of patterns from the queue.
                 * Blocks when queue size is less than n.
                 * You're responsible for deleting them once you're done.
                 *
                 * @param dest where to put the patterns
                 * @param n the number of patterns to get
                 */
                void pop(std::list<boost::shared_ptr<PatternType> >& dest, unsigned int n)
                {
                    boost::mutex::scoped_lock lock(m_mutex);
                    m_cond.wait(lock, boost::bind(&my_type::size_ge_val, this, n));

                    for (unsigned int i = 0; i < n; ++i) {
                        dest.push_back(m_queue.front());
                        m_queue.pop();
                        if(m_patterns_to_epoch_end == 0 && m_signal_restart){
                            m_patterns_to_epoch_end = -1;
                            LOG4CXX_INFO(m_log, "Dataset throws epoch_end exception");
                           throw epoch_end();
                        }else if(m_patterns_to_epoch_end > 0)
                            m_patterns_to_epoch_end--;
                    }
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
                    if(m_shuffle)
                        std::random_shuffle(m_indices.begin(), m_indices.end());
                }
               
                /**
                 * return number of images in the dataset.
                 */
                size_t size()const{
                    return m_indices.size();
                }
        };

        /**
         * Base class of methods that transform a bbtools::image_meta_info into
         * a (series of) instances of type pattern.
         *
         * All patterns generated are added to an image_queue with locking, so
         * that many image_loader instances can work in parallel.
         *
         * The meta-data that is processed by the image loader can be provided
         * e.g. by instances of the image_dataset class.
         */
        class image_loader{
            protected:
                image_queue<classification_pattern>* m_queue;               ///< where to put loaded patterns
                const bbtools::image_meta_info* m_meta;      ///< meta-infos for the image to be processed

            public:
                /**
                 * ctor.
                 * @param queue where to store patterns
                 * @param meta the meta-infos of the image to be loaded
                 */
                image_loader(image_queue<classification_pattern>* queue, const bbtools::image_meta_info* meta);
                virtual void operator()()=0;
        };

        /** 
         * Loads one image into a single pattern. 
         */
        class sample_image_loader : image_loader{
            private:
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
                sample_image_loader(image_queue<classification_pattern>* queue, 
                        const bbtools::image_meta_info* meta, 
                        unsigned int pattern_size, 
                        bool grayscale,
                        unsigned int n_classes);
                sample_image_loader(image_queue<classification_pattern>* queue, 
                        const bbtools::image_meta_info* meta);

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
                Queue* m_queue;
                Dataset* m_dataset;

                boost::thread m_pool_thread;
                bool m_running;
                bool m_request_stop;

                WorkerFactory m_worker_factory;
                log4cxx::LoggerPtr m_log;

                unsigned int m_min_pipe_len, m_max_pipe_len;

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
                    : m_queue(&queue)
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
                        unsigned int size = m_queue->size();
                        if(size < m_min_pipe_len && !m_running){
                            for (unsigned int i = 0; i < m_min_pipe_len && i + size < m_max_pipe_len; ++i)
                            {
                                m_asio_queue.post(m_worker_factory(m_queue, &m_dataset->get(cnt)));
                                cnt = (cnt+1) % m_dataset->size();
                                if(cnt == 0){
                                    LOG4CXX_INFO(m_log, "Roundtrip through dataset (" << m_dataset->size()<< ") completed. Shuffling.");
                                    m_dataset->shuffle();
                                    m_asio_queue.post(boost::bind(&Queue::on_epoch_ends, m_queue));
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
