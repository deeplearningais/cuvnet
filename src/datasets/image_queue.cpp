#include <iostream>
#include <fstream>
#include <CImg.h>
#include <log4cxx/logger.h>
#include <boost/asio.hpp>
#include <boost/foreach.hpp>

#include <cuv.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include "image_queue.hpp"

namespace{
    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("image_datasets");
}

namespace cuvnet { namespace image_datasets {

    void image_dataset::read_meta_info(std::vector<bbtools::image_meta_info>& dest, const std::string& filename){
        std::ifstream ifs(filename.c_str());
        if(!ifs) {
            LOG4CXX_ERROR(m_log, "Could not open metadata in `"<<filename<<"'");
            exit(1);
        }
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
        LOG4CXX_WARN(m_log, "MetaData loaded:" << filename << " cnt_imgs:" << cnt_imgs << " cnt_objs:" << cnt_objs);
    }

    image_dataset::image_dataset(const std::string& filename, bool shuffle){
        m_log = log4cxx::Logger::getLogger("image_dataset");
        read_meta_info(m_dataset, filename);
        for (unsigned int i = 0; i < m_dataset.size(); ++i)
            m_indices.push_back(i);
        if(shuffle)
            std::random_shuffle(m_indices.begin(), m_indices.end());
    }

    image_loader::image_loader(image_queue<pattern>* queue, const bbtools::image_meta_info* meta, const output_properties* op)
        :m_queue(queue), m_meta(meta), m_output_properties(op){}

    whole_image_loader::whole_image_loader(image_queue<pattern>* queue, 
            const bbtools::image_meta_info* meta, 
            const output_properties* op,
            unsigned int pattern_size,
            bool grayscale)
        : image_loader(queue, meta, op)
        , m_grayscale(grayscale)
        , m_pattern_size(pattern_size)
    {
        m_log = log4cxx::Logger::getLogger("whole_image_loader");
        //LOG4CXX_INFO(m_log, "started: pattern_size="<<m_pattern_size<<", grayscale="<<m_grayscale);
    }

    void whole_image_loader::operator()(){
        //LOG4CXX_INFO(m_log, "processing file `"<<m_meta->filename<<"'");
        bbtools::image bbimg(m_meta->filename);
        bbimg.meta = *m_meta;
        
        bbtools::sub_image si(bbimg);

        si.constrain_to_orig(true).extend_to_square();
        si.crop_with_padding().scale_larger_dim(m_pattern_size);

        cimg_library::CImg<unsigned char>& img = *si.pcut;
        if(m_grayscale)
            img = img.channel(0)*0.299 + img.channel(1)*0.587 + img.channel(2)*0.114;
        cuvAssert(img.width() == (int)m_pattern_size);
        cuvAssert(img.height() == (int)m_pattern_size);

        int final_size = (m_pattern_size - m_output_properties->crop_h) / m_output_properties->scale_h;
        int final_start = m_output_properties->crop_h / 2;
        cimg_library::CImg<unsigned char> tch(final_size, final_size);
        cimg_library::CImg<unsigned char> ign(final_size, final_size);

        pattern* pat_ptr = new pattern;
        pattern& pat = *pat_ptr;

        tch.fill(0);
        ign.fill(1);
        //si.mark_objects(1, 255, 0.1f, &tch, &pat.bboxes);

        {
            pat.bboxes = si.get_objects();
            float color = 255.f;
            const output_properties& op = *m_output_properties;
            BOOST_FOREACH(const bbtools::rectangle& s, pat.bboxes[0]){ // TODO only one map
                bbtools::rectangle r;
                r.ymin = std::min(final_size-1.f, std::max(0.f, (s.ymin-op.crop_h/2.f)) / (float)op.scale_h) + 0.5f;
                r.xmin = std::min(final_size-1.f, std::max(0.f, (s.xmin-op.crop_w/2.f)) / (float)op.scale_w) + 0.5f;
                r.ymax = std::min(final_size-1.f, std::max(0.f, (s.ymax-op.crop_h/2.f)) / (float)op.scale_h) + 0.5f;
                r.xmax = std::min(final_size-1.f, std::max(0.f, (s.xmax-op.crop_w/2.f)) / (float)op.scale_w) + 0.5f;
                tch.draw_rectangle(r.xmin, r.ymin,0,0, 
                        r.xmax, r.ymax, tch.depth()-1, tch.spectrum()-1, color);
            }
        }
        
        //si.mark_objects(0, 255, 0.1f, &ign);
        //si.mark_objects(0, 255, 1, &img);
        //si.fill_padding(0, &ign);

        //tch.crop(final_start, final_start, tch.width()-final_start-1, tch.height()-final_start-1, false);
        //ign.crop(final_start, final_start, ign.width()-final_start-1, ign.height()-final_start-1, false);
        //tch.resize(final_size, final_size, -100, -100, 3, 2, 0.5f, 0.5f);
        //ign.resize(final_size, final_size, -100, -100, 3, 2, 0.5f, 0.5f);

        //pat.meta_info = *meta;
        int n_dim = m_grayscale ? 1 : 3;
        pat.img.resize(cuv::extents[n_dim][m_pattern_size][m_pattern_size]);
        pat.tch.resize(cuv::extents[tch.depth()][tch.height()][tch.width()]);
        pat.ign.resize(cuv::extents[ign.depth()][ign.height()][ign.width()]);
        cuv::convert(pat.img, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[n_dim][m_pattern_size][m_pattern_size] , img.data()));
        cuv::convert(pat.tch, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[tch.depth()][tch.height()][tch.width()], tch.data()));
        cuv::convert(pat.ign, cuv::tensor<unsigned char,cuv::host_memory_space>(cuv::extents[ign.depth()][ign.height()][ign.width()], ign.data()));

        pat.img /= 255.f; 
        pat.img -= 0.5f;
        pat.ign /= 255.f;

        pat.tch /= 255.f; // in [0, 1]

        //LOG4CXX_INFO(m_log, "done processing pattern, pushing to queue");
        m_queue->push(pat_ptr);
    }


    namespace detail
    {
        struct asio_queue_impl{
            boost::asio::io_service io_service;
            boost::asio::io_service::work work;
            boost::thread_group threads;
            asio_queue_impl(unsigned int n_threads)
                :io_service(), work(io_service){
                    if(n_threads == 0)
                        n_threads = boost::thread::hardware_concurrency();
                    for (std::size_t i = 0; i < n_threads; ++i)
                            threads.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
            }
            void stop(){
                io_service.stop();
                threads.join_all();
            }
            ~asio_queue_impl(){
                this->stop();
            }
        };

        asio_queue::asio_queue(unsigned int n_threads){
            m_impl.reset(new asio_queue_impl(n_threads));
        }

        void asio_queue::stop(){
            m_impl->stop();
        }

        void asio_queue::post(boost::function<void(void)> f){
            m_impl->io_service.post(f);
        }
    }
        
} } // namespace image_datasets, cuvnet
