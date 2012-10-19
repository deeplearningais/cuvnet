#include <sys/syscall.h> /* for pid_t, syscall, SYS_gettid */
#include<iostream>
#include<fstream>
#include<queue>
#include<iterator>
#include<boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>
#include <boost/format.hpp>
#include <cuv.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <log4cxx/logger.h>

#define cimg_use_jpeg
#include <CImg.h>


#include "voc_detection.hpp"

namespace{
    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("voc_ds");
}


namespace cuvnet
{

    /**
     * loads an image from file using metadata, processes it and places it in queue
     */
    struct voc_detection_file_loader{
        boost::mutex* mutex;
        std::queue<voc_detection_dataset::pattern>* loaded_data;
        const bbtools::image_meta_info* meta;
        const voc_detection_dataset::output_properties* output_properties;
        voc_detection_file_loader(boost::mutex* m, std::queue<voc_detection_dataset::pattern>* dest, 
                const bbtools::image_meta_info* _meta,
                const voc_detection_dataset::output_properties* op)
            : mutex(m)
              , loaded_data(dest)
              , meta(_meta)
              , output_properties(op)
        {
        }
#if 1
        void split_up_bbs(bbtools::image& img, unsigned int crop_square_size){
            std::list<voc_detection_dataset::pattern> storage;
            BOOST_FOREACH(const bbtools::object& obj, meta->objects){
                if(obj.klass != 14) // 14: Person
                    continue;
                
                float w = obj.bb.xmax - obj.bb.xmin+1;
                float h = obj.bb.ymax - obj.bb.ymin+1;
                float cx = (obj.bb.xmax + obj.bb.xmin)/2.f;
                float cy = (obj.bb.ymax + obj.bb.ymin)/2.f;

                float bbscale = 2.f;
                bbtools::sub_image si(img, obj.bb);
                si.pos.xmin = cx - bbscale * w/2.f;
                si.pos.xmax = cx + bbscale * w/2.f;
                si.pos.ymin = cy - bbscale * h/2.f;
                si.pos.ymax = cy + bbscale * h/2.f;

#define STORE_IMAGES_SEQUENTIALLY 0
#if STORE_IMAGES_SEQUENTIALLY
                // store images into a local storage and put them in
                // the queue as one block after the loops
                typedef std::list<voc_detection_dataset::pattern> list_type;
                typedef voc_detection_dataset::pattern arg_type;
                ensure_square_and_enqueue(
                        boost::bind(static_cast<void (list_type::*)(const arg_type&)>(&list_type::push_back), &storage, _1), 
                        si, crop_square_size, false /* no locking */);
#else
                // store images into the queue as soon as they are ready
                // TODO: for some reason, this does not seem to work --
                //       images look like they're cut out before smoothing?
                typedef std::queue<voc_detection_dataset::pattern> queue_type;
                typedef voc_detection_dataset::pattern arg_type;

                ensure_square_and_enqueue(
                        boost::bind( static_cast<void (queue_type::*)(const arg_type&)>(&queue_type::push)
                            , loaded_data, _1),
                        si, crop_square_size, true /* locking */);
#endif
            }
#if STORE_IMAGES_SEQUENTIALLY
            boost::mutex::scoped_lock lock(*mutex);
            BOOST_FOREACH(voc_detection_dataset::pattern& pat, storage){
                loaded_data->push(pat);
            }
#endif
#undef STORE_IMAGES_SEQUENTIALLY
            //exit(1);

        }
#endif
        
#if 0
        void split_up_scales(cimg_library::CImg<unsigned char>& img, unsigned int crop_square_size){
            static const unsigned int n_scales = 3;
            cimg_library::CImg<unsigned char> pyramid[n_scales];
            std::list<voc_detection_dataset::pattern> storage;

            // size of resulting (square) patches
            int min_crop_square_overlap = crop_square_size / 4;
            int max_crop_square_overlap = crop_square_size / 2;
            
            // highest level in pyramid should have longest edge at crop_square_size
            // lower layers are half the size... 
            int pyr0_size = std::ceil(crop_square_size * pow(2.f, (float) n_scales-1));
            float orig_to_pyr0 = pyr0_size / (float) std::max(img.width(), img.height());

            // create a Gaussian pyramid
            pyramid[0] = img.get_blur(1.0f * orig_to_pyr0, 1.0f * orig_to_pyr0, 0, 1);
            pyramid[0].resize(orig_to_pyr0*img.width(), orig_to_pyr0*img.height(), -100, -100, 3);
            for(unsigned int scale=1; scale < n_scales; scale++){
                pyramid[scale] = pyramid[scale-1].get_blur(1.f, 1.f, 0, 1);
                pyramid[scale].resize(pyramid[scale].width()/2, pyramid[scale].height()/2, -100, -100, 3);
            }

            // take apart images from each scale of the pyramid
            for (unsigned int scale = 0; scale < n_scales; ++scale){
                cimg_library::CImg<unsigned char>& pimg = pyramid[scale];
                int scaled_height = pimg.height();
                int scaled_width  = pimg.width();

                // determine number of subimages in each direction assuming minimum overlap
                // this will yield a large number of subimages, which we can then relax.
                int n_subimages_y = ceil((scaled_height-min_crop_square_overlap) / (float)(crop_square_size-min_crop_square_overlap));
                int n_subimages_x = ceil((scaled_width -min_crop_square_overlap) / (float)(crop_square_size-min_crop_square_overlap));

                // determine the /actual/ overlap we get, when images are evenly distributed
                int overlap_y=0, overlap_x=0;
                if(n_subimages_y > 1)
                    overlap_y = (n_subimages_y*crop_square_size - scaled_height) / (n_subimages_y-1);
                if(n_subimages_x > 1)
                    overlap_x = (n_subimages_x*crop_square_size - scaled_width) / (n_subimages_x-1);

                // adjust number of images if overlap not good
                while(n_subimages_y > 1  &&  overlap_y > max_crop_square_overlap){
                    n_subimages_y --;
                    if(n_subimages_y == 1)
                        overlap_y = 0;
                    else
                        overlap_y = (n_subimages_y*crop_square_size - scaled_height) / (n_subimages_y-1);
                }
                while(n_subimages_x > 1  &&  overlap_x > max_crop_square_overlap){
                    n_subimages_x --;
                    if(n_subimages_x == 1)
                        overlap_x = 0;
                    else
                        overlap_x = (n_subimages_x*crop_square_size - scaled_width) / (n_subimages_x-1);
                }

                // determine the /actual/ stride we need to move at
                int stride_y = crop_square_size;
                int stride_x = crop_square_size;

                for (int sy = 0; sy < n_subimages_y; ++sy)
                {
                    // determine starting point of square in y-direction
                    int ymin = std::max(0, sy * stride_y - sy * overlap_y);
                    int ymax = ymin + crop_square_size;

                    // move towards top if growing over bottom
                    if(ymax >= pimg.height()){
                        int diff = ymax - pimg.height() - 1;
                        ymax  = std::max(0, ymax - diff);
                        ymin  = std::max(0, ymin - diff);
                        if(ymin == 0) // we hit both edges: use whole range
                            ymax = pimg.height()-1;
                    }

                    for (int sx = 0; sx < n_subimages_x; ++sx)
                    {
                        // determine starting point of square in x-direction
                        int xmin = std::max(0, sx * stride_x - sx * overlap_x);
                        int xmax = xmin + crop_square_size;

                        // move towards top if growing over bottom
                        if(xmax >= pimg.width()){
                            int diff = xmax - pimg.width() - 1;
                            xmax  = std::max(0, xmax - diff);
                            xmin  = std::max(0, xmin - diff);
                            if(xmin == 0) // we hit both edges: use whole range
                                xmax = pimg.width()-1;
                        }

                        cimg_library::CImg<unsigned char> simg = pimg.get_crop(xmin, ymin, xmax, ymax, false);

                        voc_detection_dataset::pattern pat;
                        pat.meta_info = *meta;

                        float scale_x = img.width() / (float)scaled_width;
                        float scale_y = img.height() / (float)scaled_height;

                        // remember where current crop came from
                        pat.meta_info.orig_xmin = xmin * scale_x;
                        pat.meta_info.orig_ymin = ymin * scale_y;
                        pat.meta_info.orig_xmax = xmax * scale_x;
                        pat.meta_info.orig_ymax = ymax * scale_y;

                        pat.meta_info.n_scales    = n_scales;
                        pat.meta_info.n_subimages = n_subimages_x * n_subimages_y;
                        pat.meta_info.scale_id    = scale;
                        pat.meta_info.subimage_id = sy * n_subimages_x + sx;

                        // adjust object positions according to crop
                        BOOST_FOREACH(bbtools::object& o, pat.meta_info.objects){
                            o.xmin = o.xmin/scale_x - xmin;
                            o.xmax = o.xmax/scale_x - xmin;

                            o.ymin = o.ymin/scale_y - ymin;
                            o.ymax = o.ymax/scale_y - ymin;
                        }

#define STORE_IMAGES_SEQUENTIALLY 1
#if STORE_IMAGES_SEQUENTIALLY
                        // store images into a local storage and put them in
                        // the queue as one block after the loops
                        typedef std::list<voc_detection_dataset::pattern> list_type;
                        typedef voc_detection_dataset::pattern arg_type;
                        ensure_square_and_enqueue(
                                boost::bind(static_cast<void (list_type::*)(const arg_type&)>(&list_type::push_back), &storage, _1), 
                                simg, crop_square_size, pat, false /* no locking */);

                        if(0){
                            voc_detection_dataset::pattern pat = storage.back();
                            cuv::tensor<float, cuv::host_memory_space> tmp = pat.img.copy();
                            tmp -= cuv::minimum(tmp);
                            tmp /= cuv::maximum(tmp) + 0.0001f;
                            tmp *= 255.f;
                            cuv::libs::cimg::save(tmp, boost::str(boost::format("image-s%02d-c%05d.jpg") % pat.meta_info.scale_id % pat.meta_info.subimage_id ));
                        }
#else
                        // store images into the queue as soon as they are ready
                        // TODO: for some reason, this does not seem to work --
                        //       images look like they're cut out before smoothing?
                        typedef std::queue<voc_detection_dataset::pattern> queue_type;
                        typedef voc_detection_dataset::pattern arg_type;

                        ensure_square_and_enqueue(
                                boost::bind( static_cast<void (queue_type::*)(const arg_type&)>(&queue_type::push)
                                    , loaded_data, _1),
                                img, crop_square_size, pat, true /* locking */);
#endif
                    }
                }
            }
#if STORE_IMAGES_SEQUENTIALLY
            boost::mutex::scoped_lock lock(*mutex);
            BOOST_FOREACH(voc_detection_dataset::pattern& pat, storage){
                loaded_data->push(pat);
            }
#endif
            //exit(1);

        }
#endif

        template<class T>
        void ensure_square_and_enqueue(T output, bbtools::sub_image& si, unsigned int sq_size, bool lock){

            //unsigned int n_classes = 1;
            si.constrain_to_orig(true).extend_to_square();
            bool found_obj = si.has_objects();
            if(!found_obj)
                return;
            si.crop_with_padding().scale_larger_dim(sq_size);
            cimg_library::CImg<unsigned char>& img = *si.pcut;
            cuvAssert(img.width() == (int)sq_size);
            cuvAssert(img.height() == (int)sq_size);

            int final_size = (sq_size - output_properties->crop_h) / output_properties->scale_h;
            int final_start = output_properties->crop_h / 2;
            cimg_library::CImg<unsigned char> tch(sq_size, sq_size);
            cimg_library::CImg<unsigned char> ign(sq_size, sq_size);
            
            tch.fill(0);
            ign.fill(0);
            si.mark_objects(2, 255, 0.25f, &tch);
            si.mark_objects(2, 255, 1.f, &ign);
            //si.mark_objects(0, 255, 1, &img);
            si.fill_padding(0, &ign);

            tch.crop(final_start, final_start, tch.width()-final_start-1, tch.height()-final_start-1, false);
            ign.crop(final_start, final_start, ign.width()-final_start-1, ign.height()-final_start-1, false);
            tch.resize(final_size, final_size, -100, -100, 3, 2, 0.5f, 0.5f);
            ign.resize(final_size, final_size, -100, -100, 3, 2, 0.5f, 0.5f);
            
            // TODO: Blur now?

            // convert to cuv
            //img.RGBtoYCbCr();
            voc_detection_dataset::pattern pat;
            //pat.meta_info = *meta;
            pat.img.resize(cuv::extents[3][sq_size][sq_size]);
            pat.tch.resize(cuv::extents[tch.depth()][tch.height()][tch.width()]);
            pat.ign.resize(cuv::extents[ign.depth()][ign.height()][ign.width()]);
            cuv::convert(pat.img, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[3][sq_size][sq_size] , img.data()));
            cuv::convert(pat.tch, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[tch.depth()][tch.height()][tch.width()], tch.data()));
            cuv::convert(pat.ign, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[ign.depth()][ign.height()][ign.width()], ign.data()));

            pat.img /= 255.f;
            pat.img -= 0.5f;
            pat.ign /= 255.f;

            // set ignore to a non-zero value where teacher is "no object".
            // this reweights errors, so that it is easier to produce a
            // "positive" output.
            //cuv::tensor<unsigned char, cuv::host_memory_space> idx(pat.tch.shape());
            //cuv::apply_scalar_functor(idx, pat.tch, cuv::SF_LT, 176);
            //cuv::apply_scalar_functor(pat.ign, cuv::SF_MIN, 0.001f, &idx);

#if 1
            pat.tch /= 255.f;
            //pat.tch /= 127.f;
            //pat.tch -=   1.f;
#else
            pat.tch /= 255.f;
#endif

            // put in pipe
            if(lock) {
                boost::mutex::scoped_lock lock(*mutex);
                //loaded_data->push(pat);
                output(pat);
            }else
                output(pat);
        }


        void operator()(){
            bbtools::image img(meta->filename);
            img.meta = *meta;

            if(0){
                typedef std::queue<voc_detection_dataset::pattern> queue_type;
                typedef voc_detection_dataset::pattern arg_type;

                bbtools::sub_image si(img);
                
                ensure_square_and_enqueue(
                        boost::bind( static_cast<void (queue_type::*)(const arg_type&)>(&queue_type::push)
                            , loaded_data, _1),
                        si, 176, true);
            }else{
                // split up the image into multiple scales, extract images with
                // size 128x128 on all scales, and enqueue them.
                //split_up_scales(img, 176);
                split_up_bbs(img, 176);
            }
        }
    };
    
    /**
     * loads data from disk using meta-data (asynchronously)
     */
    class voc_detection_pipe{
        private:
            mutable boost::mutex m_loaded_data_mutex; /// workers use this to ensure thread-safe access to m_loaded_data.

            const std::vector<bbtools::image_meta_info>& dataset;
            const voc_detection_dataset::output_properties& m_output_properties;

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
                    const std::vector<bbtools::image_meta_info>& ds,
                    const voc_detection_dataset::output_properties& op,
                    unsigned int n_threads=0, unsigned int min_pipe_len=32, unsigned int max_pipe_len=0)
                : dataset(ds)
                , m_output_properties(op)
                , m_min_pipe_len(min_pipe_len)
                , m_max_pipe_len(max_pipe_len > min_pipe_len ? max_pipe_len : 3 * min_pipe_len)
                , m_n_threads(n_threads)
                , m_request_stop(false)
                , m_running(false)
            {
                if(m_n_threads == 0)
                   m_n_threads = boost::thread::hardware_concurrency();
                //m_n_threads = 1;
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
                if(m_min_pipe_len < n){
                    m_min_pipe_len = n;
                    m_max_pipe_len = 3 * n;
                }
                // TODO: use boost condition_variable!
                while(size() < n)
                    boost::this_thread::sleep(boost::posix_time::millisec(10));

                // TODO: when getting lock fails, loop again above!
                //       that way, multiple clients can use the same queue
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
                std::vector<unsigned int> idxs(dataset.size());
                for(unsigned int i=0;i<dataset.size();i++)
                    idxs[i] = i;

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
                                        &dataset[idxs[cnt]],
                                        &m_output_properties));
                            cnt = (cnt+1) % dataset.size();
                            if(cnt == 0) {
                                LOG4CXX_INFO(g_log, "Roundtrip through dataset completed. Shuffling.");
                                std::random_shuffle(idxs.begin(), idxs.end());
                            }
                        }
                        grp.join_all();
                    }
                    boost::this_thread::sleep(boost::posix_time::millisec(10));
                }
            }
    };

    void voc_detection_dataset::set_output_properties(
            unsigned int scale_h, unsigned int scale_w,
            unsigned int crop_h, unsigned int crop_w){
        m_output_properties.scale_h = scale_h;
        m_output_properties.scale_w = scale_w;
        m_output_properties.crop_h = crop_h;
        m_output_properties.crop_w = crop_w;
    }
    void voc_detection_dataset::switch_dataset(voc_detection_dataset::subset ss, int n_threads){

        n_threads = n_threads == 0 ? m_n_threads : n_threads;

        if(m_pipe)
            m_pipe->request_stop();

        switch(ss){
            case SS_TRAIN:
                m_pipe.reset(new voc_detection_pipe(m_training_set, m_output_properties, n_threads));
                break;
            case SS_VAL:
                m_pipe.reset(new voc_detection_pipe(m_val_set, m_output_properties, n_threads));
                break;
            case SS_TEST:
                m_pipe.reset(new voc_detection_pipe(m_test_set, m_output_properties, n_threads));
                break;
            default:
                throw std::runtime_error("VOC Detection Dataset: cannot switch to supplied subset!");
        }
        m_pipe->start();
    }

    voc_detection_dataset::voc_detection_dataset(const std::string& train_filename, const std::string& test_filename, int n_threads, bool verbose)
    {
        read_meta_info(m_training_set, train_filename, verbose);
        read_meta_info(m_test_set, test_filename, verbose);
        srand(time(NULL));
        std::random_shuffle(m_training_set.begin(), m_training_set.end());
        //switch_dataset(SS_TRAIN);
    }

    void voc_detection_dataset::get_batch(std::list<pattern>& dest, unsigned int n){
        m_pipe->get_batch(dest, n);
    }
    unsigned int voc_detection_dataset::size_available()const{
        return m_pipe->size();
    }

    void voc_detection_dataset::read_meta_info(std::vector<bbtools::image_meta_info>& dest, const std::string& filename, bool verbose){
        std::ifstream ifs(filename.c_str());
        unsigned int n_objs;
        unsigned int cnt_imgs=0, cnt_objs=0;
        while(ifs){
            bbtools::image_meta_info imi;
            ifs >> imi.filename;
            if(imi.filename.size() == 0)
                break;
            ifs >> n_objs;
            for (unsigned int i = 0; i < n_objs; ++i)
            {
                bbtools::object o;
                ifs >> o.klass;
                ifs >> o.truncated;
                ifs >> o.bb.xmin >> o.bb.xmax >> o.bb.ymin >> o.bb.ymax;
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
    void voc_detection_dataset::save_results(std::list<pattern>& results){
        BOOST_FOREACH(pattern& pat, results){
            m_return_queue.push_back(pat);

            if(pat.meta_info.scale_id != pat.meta_info.n_scales - 1)
                continue;
            if(pat.meta_info.subimage_id != pat.meta_info.n_subimages - 1)
                continue;

            // the image has been completely processed. Collect all cropped parts.
            std::list<pattern> ready_image;

            // move patterns from m_return_queue to ready_image w/o copying (using splice)
            // http://stackoverflow.com/questions/501962/erasing-items-from-an-stl-list
            std::list<pattern>::iterator it = m_return_queue.begin();
            while (it != m_return_queue.end()) {
                bool filename_matches = it->meta_info.filename == pat.meta_info.filename;
                if (filename_matches)
                {
                    std::list<pattern>::iterator old_it = it++;
                    ready_image.splice(ready_image.end(), m_return_queue, old_it);
                }
                else
                    ++it;
            }

            // TODO: dispatch resulting image
            //std::cout << "Done with  image `" << pat.meta_info.filename << "'" << std::endl;
        }
    }
}
