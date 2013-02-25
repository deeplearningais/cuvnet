#ifndef __IMAGE_QUEUE_HPP__
#     define __IMAGE_QUEUE_HPP__

#include <queue>
#include <boost/thread.hpp>
#include <cuvnet/tools/logging.hpp>
#include <datasets/bounding_box_tools.hpp>

namespace cuvnet
{

    namespace image_datasets
    {
        /// a fully loaded pattern.                                                                                                                                     
        struct pattern{                                                                                                                                                 
            bbtools::image_meta_info meta_info;                                                                                                                         
            cuv::tensor<float,cuv::host_memory_space> img; ///< image                                                                                                   
            cuv::tensor<float,cuv::host_memory_space> tch; ///< teacher                                                                                                 
            cuv::tensor<float,cuv::host_memory_space> ign; ///< ignore mask                                                                                             

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
                mutable boost::mutex m_mutex;
                std::queue<PatternType*> m_queue;
                log4cxx::LoggerPtr m_log;
            public:
                image_queue(){
                    m_log = log4cxx::Logger::getLogger("image_queue");
                }

                void push(PatternType* pat, bool lock=true){ 
                    if(!lock){
                        m_queue.push(pat); 
                        return;
                    }
                    boost::mutex::scoped_lock l(m_mutex);
                    m_queue.push(pat); 
                }

                /// return the number of patterns currently stored in the queue
                size_t size()const{
                    return m_queue.size();
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
                void pop(std::list<PatternType*>& dest, unsigned int n)
                {
                    // TODO: use boost condition_variable!                                                                                                                      
                    while(size() < n)                                                                                                                                           
                    {
                        //LOG4CXX_WARN(m_log, "queue size="<<size()<<", but requested "<<n<<" patterns-->sleeping");
                        boost::this_thread::sleep(boost::posix_time::millisec(10));                                                                                             
                    }

                    // TODO: when getting lock fails, loop again above!                                                                                                         
                    //       that way, multiple clients can use the same queue                                                                                                  
                    boost::mutex::scoped_lock lock(m_mutex);
                    for (unsigned int i = 0; i < n; ++i) {                                                                                                                      
                        dest.push_back(m_queue.front());                                                                                                                  
                        m_queue.pop();                                                                                                                                    
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
                 * return number of images in the dataset.
                 */
                size_t size()const{
                    return m_indices.size();
                }
        };

        /** 
         * assuming the input tensors are of shape NxN, the teacher/ignore
         * tensors are of shape MxM, where M = (N-crop)/scale.
         * This is useful if the teacher/ignore variables are to be used in
         * a network that performs valid convolutions (crop) and pooling
         * (scale) operations.
         */
        struct output_properties{
            unsigned int scale_h, scale_w, crop_h, crop_w;
            output_properties(unsigned int scale_h_, unsigned int scale_w_, unsigned int crop_h_, unsigned int crop_w_)
                :scale_h(scale_h_), scale_w(scale_w_), crop_h(crop_h_), crop_w(crop_w_)
            {}
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
                image_queue<pattern>* m_queue;               ///< where to put loaded patterns
                const bbtools::image_meta_info* m_meta;      ///< meta-infos for the image to be processed
                const output_properties* m_output_properties; ///< how outputs should be parameterized

            public:
                /**
                 * ctor.
                 * @param queue where to store patterns
                 * @param meta the meta-infos of the image to be loaded
                 * @param op how outputs should be parameterized
                 */
                image_loader(image_queue<pattern>* queue, const bbtools::image_meta_info* meta, const output_properties* op);
                virtual void operator()()=0;
        };

        /** 
         * Loads one image into a single pattern. 
         */
        class whole_image_loader : image_loader{
            private:
                bool m_grayscale;
                unsigned int m_pattern_size;
                log4cxx::LoggerPtr m_log;
            public:
                /**
                 * @param queue where to put the results
                 * @param meta the image to load
                 * @param output_properties how the output parts of the pattern should be cropped/scaled w.r.t. the input
                 * @param pattern_size width/height of patterns
                 * @param grayscale if true, discard color information
                 */
                whole_image_loader(image_queue<pattern>* queue, 
                        const bbtools::image_meta_info* meta, 
                        const output_properties* op,
                        unsigned int pattern_size, 
                        bool grayscale);

                virtual void operator()();
        };
        /**
         * starts many workers which process images to patterns and enqueue them into an image queue.
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
                Queue* m_queue;
                unsigned int m_n_threads;
                Dataset* m_dataset;

                boost::thread m_pool_thread;
                bool m_running;
                bool m_request_stop;

                WorkerFactory m_worker_factory;
                log4cxx::LoggerPtr m_log;

                unsigned int m_min_pipe_len, m_max_pipe_len;
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
                    , m_n_threads(n_threads)
                    , m_dataset(&ds)
                    , m_running(false)
                    , m_request_stop(false)
                    , m_worker_factory(worker_factory)
                    , m_min_pipe_len(min_queue_len)
                    , m_max_pipe_len(max_queue_len)
                {
                    if(n_threads == 0)
                        m_n_threads = boost::thread::hardware_concurrency();

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
                    m_running = true;
                    while(!m_request_stop){
                        unsigned int size = m_queue->size();
                        if(size < m_min_pipe_len){
                            boost::thread_group grp;
                            for (unsigned int i = 0; i < m_n_threads && i+size < m_max_pipe_len; ++i)
                            {
                                grp.create_thread(m_worker_factory(
                                            m_queue, 
                                            &m_dataset->get(cnt)));
                                cnt = (cnt+1) % m_dataset->size();
                                //if(cnt == 0)
                                //    LOG4CXX_INFO(g_log, "Roundtrip through dataset completed. Shuffling.");
                            }
                            grp.join_all();
                        }
                        boost::this_thread::sleep(boost::posix_time::millisec(10));
                    }
                    m_running = false;
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
