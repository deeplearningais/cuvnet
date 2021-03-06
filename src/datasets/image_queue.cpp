#include <iostream>
#include <fstream>
#include <CImg.h>
#include <log4cxx/logger.h>
#include <boost/asio.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <cuv.hpp>
//#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "image_queue.hpp"

namespace{
    log4cxx::LoggerPtr g_log = log4cxx::Logger::getLogger("image_datasets");
}

namespace cuvnet { namespace image_datasets {

    void dstack_mat2tens(cuv::tensor<float, cuv::host_memory_space>& tens,
            const std::vector<cv::Mat>& src, bool reverse=false){
        assert(src.size() >= 1);
        int d = src.size();
        int h = src.front().rows;
        int w = src.front().cols;

        tens.resize(cuv::extents[d][h][w]);
        for (int i = 0; i < d; ++i)
        {
            int other = reverse ? d-1-i : i;
            typedef unsigned char src_dtype;
            cuv::tensor<src_dtype, cuv::host_memory_space> wrapped(cuv::extents[h][w], (src_dtype*) src[other].data);
            cuv::tensor<float, cuv::host_memory_space> view = cuv::tensor_view<float, cuv::host_memory_space>(tens, cuv::indices[i][cuv::index_range()][cuv::index_range()]);
            cuv::convert(view, wrapped);
            //cv::Mat wrapped(w, h, CV_32F, &tens(i, 0, 0), CV_AUTOSTEP);
            //src[i].convertTo(wrapped, CV_32F);
        }
    }

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
            if(!ifs)
                break;
            if(imi.filename.size() == 0)
                break;
            if(imi.filename[0] == '#')
                continue;
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
        LOG4CXX_INFO(m_log, "MetaData loaded:" << filename << " cnt_imgs:" << cnt_imgs << " cnt_objs:" << cnt_objs);
    }

    image_dataset::image_dataset(const std::string& filename, bool shuffle){
        m_log = log4cxx::Logger::getLogger("image_dataset");
        LOG4CXX_INFO(m_log, "MetaData:" << filename << " created; shuffle: "<< shuffle);
        read_meta_info(m_dataset, filename);
        for (unsigned int i = 0; i < m_dataset.size(); ++i)
            m_indices.push_back(i);
        m_shuffle = shuffle;
        this->shuffle();
    }


    sample_image_loader::sample_image_loader(patternset_ptr dst, 
            const bbtools::image_meta_info* meta, 
            unsigned int pattern_size,
            bool grayscale,
            unsigned int n_classes
            )
        : m_dst(dst), m_meta(meta)
        , m_grayscale(grayscale)
        , m_pattern_size(pattern_size)
        , m_n_classes(n_classes)
    {
        m_log = log4cxx::Logger::getLogger("sample_image_loader");
        //LOG4CXX_INFO(m_log, "started: pattern_size="<<m_pattern_size<<", grayscale="<<m_grayscale);
    }

    void sample_image_loader::operator()(){
        bbtools::image bbimg(m_meta->filename);
        bbimg.meta = *m_meta;

        bbtools::sub_image si(bbimg);

        si.constrain_to_orig(true).extend_to_square();
        si.crop_with_padding().scale_larger_dim(m_pattern_size);

        cv::Mat& img = *si.pcut;

        if(m_grayscale)
            cvtColor(img,img,CV_RGB2GRAY);

        cuvAssert(img.size.p[0] == (int)m_pattern_size);
        cuvAssert(img.size.p[1] == (int)m_pattern_size);

        boost::shared_ptr<classification_pattern> pat_ptr(new classification_pattern);
        classification_pattern& pat = *pat_ptr;

        int n_dim = m_grayscale ? 1 : 3;
        pat.img.resize(cuv::extents[n_dim][m_pattern_size][m_pattern_size]);

        std::vector<cv::Mat> img_vec;
        cv::split(img, img_vec);

        dstack_mat2tens(pat.img, img_vec, true); // reverse rgb here
        pat.tch.resize(m_n_classes);
        pat.tch = 0.f;
        BOOST_FOREACH(const cuvnet::bbtools::object& o, m_meta->objects){
            pat.tch(o.klass) = 1.f;
        }

        pat.img /= 255.f; 
        pat.img -= 0.5f;
        m_dst->push(pat_ptr, true);
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

        asio_queue::asio_queue(unsigned int n_threads)
            :m_n_threads(n_threads)
        {
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
