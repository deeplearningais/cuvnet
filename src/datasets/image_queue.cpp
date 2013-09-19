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
            const std::vector<cv::Mat>& src){
        assert(src.size() >= 1);
        int d = src.size();
        int h = src.front().rows;
        int w = src.front().cols;

        tens.resize(cuv::extents[d][h][w]);
        for (int i = 0; i < d; ++i)
        {
            cuv::tensor<unsigned char, cuv::host_memory_space> wrapped(cuv::extents[h][w], src[i].data);
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
            bool grayscale,
            unsigned int n_classes
            )
        : image_loader(queue, meta, op)
        , m_grayscale(grayscale)
        , m_pattern_size(pattern_size)
        , m_n_classes(n_classes)
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

        cv::Mat& img = *si.pcut;
        if(m_grayscale)
            //img = img.channel(0)*0.299 + img.channel(1)*0.587 + img.channel(2)*0.114;
            cvtColor(img,img,CV_RGB2GRAY);
        cuvAssert(img.size.p[0] == (int)m_pattern_size);
        cuvAssert(img.size.p[1] == (int)m_pattern_size);

        int final_size = (m_pattern_size - m_output_properties->crop_h) / m_output_properties->scale_h;

        std::vector<cv::Mat> tch_vec;
        std::vector<cv::Mat> ign_vec;

        pattern* pat_ptr = new pattern;
        pattern& pat = *pat_ptr;

        //si.mark_objects(1, 255, 0.1f, &tch, &pat.bboxes);

        {
            pat.bboxes = si.get_objects(m_n_classes);
            const output_properties& op = *m_output_properties;
            for(unsigned int map=0; map != pat.bboxes.size(); map++){
                cv::Mat tch = cv::Mat::zeros(final_size, final_size, CV_8U);
                cv::Mat ign = cv::Mat::zeros(final_size, final_size, CV_8U);

                BOOST_FOREACH(const bbtools::rectangle& s, pat.bboxes[map]){ // TODO only one map
                    bbtools::rectangle r = s.scale(0.5f);
                    op.transform(r.xmin, r.ymin);
                    op.transform(r.xmax, r.ymax);
                    r.ymin = std::min(final_size-1, std::max(0, r.ymin));
                    r.xmin = std::min(final_size-1, std::max(0, r.xmin));
                    r.ymax = std::min(final_size-1, std::max(0, r.ymax));
                    r.xmax = std::min(final_size-1, std::max(0, r.xmax));
                    //tch.draw_rectangle(r.xmin, r.ymin,0,0, 
                            //r.xmax, r.ymax, tch.depth()-1, tch.spectrum()-1, color);
                    cv::rectangle(tch, cv::Point(r.xmin, r.ymin), 
                            cv::Point(r.xmax, r.ymax), cv::Scalar(1u), CV_FILLED);
                    cv::rectangle(ign, cv::Point(r.xmin-1, r.ymin-1), 
                            cv::Point(r.xmax+1, r.ymax+1), cv::Scalar(1u), CV_FILLED);
                    //ign.draw_rectangle(r.xmin-1, r.ymin-1, r.xmax+1, r.ymax+1, &clr, 1.f, ~0U);
                    //ign.draw_rectangle(r.xmin+0, r.ymin+0, r.xmax+0, r.ymax+0, &clr, 1.f, ~0U);
                    //ign.draw_rectangle(r.xmin+1, r.ymin+1, r.xmax-1, r.ymax-1, &clr, 1.f, ~0U);
                }
                si.fill_padding(0, &ign, *m_output_properties);
                tch_vec.push_back(tch);
                ign_vec.push_back(ign);
            }
        }
        
        //si.mark_objects(0, 255, 0.1f, &ign);
        //si.mark_objects(0, 255, 1, &img);


        //tch.crop(final_start, final_start, tch.width()-final_start-1, tch.height()-final_start-1, false);
        //ign.crop(final_start, final_start, ign.width()-final_start-1, ign.height()-final_start-1, false);
        //tch.resize(final_size, final_size, -100, -100, 3, 2, 0.5f, 0.5f);
        //ign.resize(final_size, final_size, -100, -100, 3, 2, 0.5f, 0.5f);

        //pat.meta_info = *meta;
        int n_dim = m_grayscale ? 1 : 3;
        pat.img.resize(cuv::extents[n_dim][m_pattern_size][m_pattern_size]);
        pat.tch.resize(cuv::extents[tch_vec.size()][final_size][final_size]);
        pat.ign.resize(cuv::extents[ign_vec.size()][final_size][final_size]);

        std::vector<cv::Mat> img_vec;
        cv::split(img, img_vec);

        dstack_mat2tens(pat.img, img_vec);
        dstack_mat2tens(pat.tch, tch_vec);
        dstack_mat2tens(pat.ign, ign_vec);

        pat.img /= 255.f; 
        pat.img -= 0.5f;
        //pat.ign /= 255.f;

        //pat.tch /= 255.f; // in [0, 1]

        //static int cnt = 0;
        //img.save_jpeg((boost::format("/tmp/ii/loaded_%d.jpg")%cnt++).str().c_str());


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
