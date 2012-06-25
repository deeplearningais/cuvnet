#include<iostream>
#include<fstream>
#include<queue>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>

#define cimg_use_jpeg
#include <CImg.h>

#include "voc_detection.hpp"


namespace cuvnet
{

    void square(cimg_library::CImg<unsigned char>& orig, unsigned int sq_size, voc_detection_dataset::image_meta_info& meta){
        unsigned int orig_rows = orig.height();
        unsigned int orig_cols = orig.width();

        unsigned int new_rows = orig.height() >= orig.width()  ? sq_size : orig.height()/(float)orig.width() * sq_size;
        unsigned int new_cols = orig.width() >= orig.height() ? sq_size : orig.width()/(float)orig.height() * sq_size;

        // downsample
        orig.resize(new_cols, new_rows, -100/* z */, -100 /* c */, 3 /* 3: linear interpolation */);

        // square
        orig.resize(sq_size, sq_size, -100, -100, 0 /* no interpolation */, 1 /* 0: zero border, 1: nearest neighbor */, 0.5f, 0.5f);

        float stretchx = new_cols / (float)orig_cols;
        float stretchy = new_rows / (float)orig_rows;
        float offx     = 0.5f * stretchx * std::max(0, (int)orig_rows - (int)orig_cols);
        float offy     = 0.5f * stretchy * std::max(0, (int)orig_cols - (int)orig_rows);
        std::cout << "stretchx:" << stretchx << " stretchy:" << stretchy << " offx:" << offx << " offy:" << offy << std::endl;

        unsigned int cnt=1;
        BOOST_FOREACH(voc_detection_dataset::object& o, meta.objects){
            o.xmin = stretchx * o.xmin + offx;
            o.ymin = stretchy * o.ymin + offy;
            o.xmax = std::min((float)orig.width()-1, stretchx * o.xmax + offx);
            o.ymax = std::min((float)orig.height()-1, stretchy * o.ymax + offy);
            assert(o.xmin < o.xmax);
            assert(o.ymin < o.ymax);
            assert(o.xmin >= 0);
            assert(o.ymin >= 0);
            assert(o.xmax < sq_size);
            assert(o.ymin < sq_size);
            orig.draw_rectangle(o.xmin, o.ymin, 0, 0, o.xmax, o.ymax, 0, 2, 255, std::min(0.8, 0.2 * cnt++));
        }

    }


    /**
     * loads an image from file using metadata, processes it and places it in queue
     */
    struct voc_detection_file_loader{
        boost::mutex* mutex;
        std::queue<voc_detection_dataset::pattern>* loaded_data;
        const voc_detection_dataset::image_meta_info* meta;
        voc_detection_file_loader(boost::mutex* m, std::queue<voc_detection_dataset::pattern>* dest, const voc_detection_dataset::image_meta_info* _meta)
        : mutex(m)
        , loaded_data(dest)
        , meta(_meta)
        {
        }
        void operator()(){
            voc_detection_dataset::pattern pat;
            pat.meta_info = *meta;
            // TODO: load pat using meta
            cimg_library::CImg<unsigned char> img;
            img.load_jpeg(meta->filename.c_str());
            square(img, 172, pat.meta_info);
            
            boost::mutex::scoped_lock lock(*mutex);

            img.display();

            loaded_data->push(pat);
        }
    };
    
    /**
     * loads data from disk using meta-data (asynchronously)
     */
    class voc_detection_pipe{
        private:
            mutable boost::mutex m_loaded_data_mutex; /// workers use this to ensure thread-safe access to m_loaded_data.

            const std::vector<voc_detection_dataset::image_meta_info>& dataset;

            std::queue<voc_detection_dataset::pattern> m_loaded_data;
            
            unsigned int m_min_pipe_len; ///< if pipe is shorter than this, start threads to make it grow
            unsigned int m_max_pipe_len; ///< if pipe is longer than this, stop filling it
            unsigned int m_n_threads;    ///< number of worker threads to be spawned
            bool m_request_stop; ///< if true, main loop stops
            bool m_running; ///< true between start() and request_stop()

            boost::thread m_thread; ///< can be used to stop pipe externally

        public:

            /**
             * constructor.
             *
             * @param ds dataset metadata
             * @param n_threads if 0, use number of CPUs
             * @param min_pipe_len when to start filling pipe
             * @param max_pipe_len when to stop filling pipe
             */
            voc_detection_pipe(
                    const std::vector<voc_detection_dataset::image_meta_info>& ds,
                    unsigned int n_threads=0, unsigned int min_pipe_len=32, unsigned int max_pipe_len=0)
                : dataset(ds)
                , m_min_pipe_len(min_pipe_len)
                , m_max_pipe_len(max_pipe_len > min_pipe_len ? max_pipe_len : 3 * min_pipe_len)
                , m_n_threads(n_threads)
                , m_request_stop(false)
                , m_running(false)
            {
                if(m_n_threads == 0)
                    m_n_threads = boost::thread::hardware_concurrency();
            }

            /**
             * destructor, stops thread.
             */
            ~voc_detection_pipe(){
                if(m_running)
                    request_stop();
            }

            /**
             * call this externally to stop pipe thread
             */
            void request_stop(){ 
                m_request_stop = true; 
                m_thread.join();
                m_running = false;
            }

            void get_batch(std::list<voc_detection_dataset::pattern>& dest, unsigned int n){
                // TODO: use boost condition_variable!
                while(size() < n)
                    boost::this_thread::sleep(boost::posix_time::millisec(10));

                boost::mutex::scoped_lock lock(m_loaded_data_mutex);
                for (unsigned int i = 0; i < n; ++i) {
                    dest.push_back(m_loaded_data.front());
                    m_loaded_data.pop();
                }
            }

            /** start filling the queue  of ready-to-use elements */
            void start(){
                m_thread =  boost::thread(boost::ref(*this));
                m_running = true;
            }

            /** return the number of ready-to-use data elements */
            unsigned int size()const{
                boost::mutex::scoped_lock lock(m_loaded_data_mutex);
                return m_loaded_data.size();
            }

            
            /** main loop, ensures queue stays filled and stops when \c m_request_stop is true */
            void operator()(){
                unsigned int cnt = 0;
                while(!m_request_stop){

                    unsigned int size = this->size();

                    if(size < m_min_pipe_len){
                        boost::thread_group grp;
                        for (unsigned int i = 0; i < m_n_threads && i+size < m_max_pipe_len; ++i)
                        {
                            grp.create_thread(
                                    voc_detection_file_loader(
                                        &m_loaded_data_mutex,
                                        &m_loaded_data,
                                        &dataset[cnt]));
                            cnt = (cnt+1) % dataset.size();
                        }
                        grp.join_all();
                    }
                    boost::this_thread::sleep(boost::posix_time::millisec(10));
                }
            }
    };

    void voc_detection_dataset::switch_dataset(voc_detection_dataset::subset ss){
        if(m_pipe)
            m_pipe->request_stop();

        switch(ss){
            case SS_TRAIN:
                m_pipe.reset(new voc_detection_pipe(m_training_set));
                break;
            case SS_VAL:
                m_pipe.reset(new voc_detection_pipe(m_val_set));
                break;
            case SS_TEST:
                m_pipe.reset(new voc_detection_pipe(m_test_set));
                break;
            default:
                throw std::runtime_error("VOC Detection Dataset: cannot switch to supplied subset!");
        }
        m_pipe->start();
    }

    voc_detection_dataset::voc_detection_dataset(const std::string& train_filename, const std::string& test_filename, bool verbose)
    {
        read_meta_info(m_training_set, train_filename, verbose);
        read_meta_info(m_test_set, test_filename, verbose);
        switch_dataset(SS_TRAIN);
    }

    void voc_detection_dataset::get_batch(std::list<pattern>& dest, unsigned int n){
        m_pipe->get_batch(dest, n);
    }
    unsigned int voc_detection_dataset::size_available()const{
        return m_pipe->size();
    }

    void voc_detection_dataset::read_meta_info(std::vector<image_meta_info>& dest, const std::string& filename, bool verbose){
        std::ifstream ifs(filename.c_str());
        unsigned int n_objs;
        unsigned int cnt_imgs=0, cnt_objs=0;
        while(ifs){
            image_meta_info imi;
            ifs >> imi.filename;
            if(imi.filename.size() == 0)
                break;
            ifs >> n_objs;
            for (unsigned int i = 0; i < n_objs; ++i)
            {
                object o;
                ifs >> o.klass;
                ifs >> o.truncated;
                ifs >> o.xmin >> o.xmax >> o.ymin >> o.ymax;
                imi.objects.push_back(o);
                cnt_objs ++;
            }
            dest.push_back(imi);
            cnt_imgs ++;
        }
        if(verbose){
            std::cout << "VOC:" << filename << " cnt_imgs:" << cnt_imgs << " cnt_objs:" << cnt_objs << std::endl;
        }
    }
}
