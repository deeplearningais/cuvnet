#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <third_party/cnpy/cnpy.h>
#include <cuvnet/tools/logging.hpp>
#include <fstream>

#include "cv_datasets.hpp"

namespace
{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("cv_ds"));
}

namespace datasets
{
    rotated_rect& rotated_rect::operator=(const cv::RotatedRect& r){
        x = r.center.x;
        y = r.center.y;
        h = r.size.height;
        w = r.size.width;
        a = r.angle;
        return *this;
    }

    rotated_rect::operator cv::RotatedRect()const{
        cv::RotatedRect r;
        r.center = cv::Point2f(x, y);
        r.size = cv::Size(w, h); // TODO order??
        r.angle = a;
        return r;
    }
    
    struct rgb_image{
        size_t ID;
        cv::Mat rgb;
    };

    struct rgbd_image : public virtual rgb_image{
        cv::Mat depth, height;
    };

    struct rgbt_image : public virtual rgb_image{
        std::map<int, cv::Mat> cls;
        cv::Mat ign;
        cv::vector<cv::Mat> prediction;
    };

    struct rgbdt_image : public rgbt_image, public rgbd_image{
    };

    template<>
    boost::shared_ptr<meta_data<rgb_classification_tag>::input_t> load_image(const meta_data<rgb_classification_tag>& meta){
        auto img = boost::make_shared<rgb_image>();
        img->rgb = cv::imread(meta.rgb_filename, CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(img->rgb, img->rgb, CV_BGR2RGB);
        return img;
    }
    template<>
    boost::shared_ptr<meta_data<rgb_objclassseg_tag>::input_t> load_image(const meta_data<rgb_objclassseg_tag>& meta){
        auto img = boost::make_shared<rgbt_image>();
        img->rgb = cv::imread(meta.rgb_filename, CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(img->rgb, img->rgb, CV_BGR2RGB);
        
        img->ign = cv::Mat::ones(img->rgb.rows, img->rgb.cols, CV_32FC1);
        cv::Mat m = cv::imread(meta.teacher_filename.c_str(), -1);
        unsigned short* p = (unsigned short*)m.data;
        for(unsigned int y=0; y<(unsigned int)m.rows; y++){
            for(unsigned int x=0; x<(unsigned int)m.cols; x++){
                int v = *p;

                // check if class encountered before
                if(img->cls.count(v) == 0)
                    img->cls[v] = cv::Mat::zeros(img->rgb.rows, img->rgb.cols, CV_32FC1);

                if(v == 255)
                    img->ign.at<float>(y,x) = 0;
                else
                    img->cls[v].at<float>(y,x) = 1;
                p++;
            }
        }
        return img;
    }
    
    template<>
    boost::shared_ptr<meta_data<rgbd_objclassseg_tag>::input_t> load_image(const meta_data<rgbd_objclassseg_tag>& meta){
        auto img = boost::make_shared<rgbdt_image>();
        {
            meta_data<rgb_objclassseg_tag> tmp_meta;
            tmp_meta.rgb_filename = meta.rgb_filename;
            tmp_meta.teacher_filename = meta.teacher_filename;
            auto tmp = load_image(tmp_meta);
            img->rgb = tmp->rgb;
            img->cls = tmp->cls;
            img->ign = tmp->ign;
        }

        cnpy::NpyArray npDepth = cnpy::npy_load(meta.depth_filename);
        img->depth = cv::Mat (npDepth.shape[1], npDepth.shape[0], CV_32F);
        assert(img->depth.isContinuous());

        memcpy(img->depth.data, npDepth.data, sizeof(float)*img->depth.rows*img->depth.cols);
        cv::transpose(img->depth, img->depth);
        assert(img->depth.isContinuous());
        npDepth.destruct(); // remove numpy array from memory
        
        //std::ifstream ifs(meta.depth_filename.substr(0, meta.depth_filename.size()-3) + "height");
        //img->height = cv::Mat(480, 640, CV_32FC1);
        //ifs.read((char*)img->height.data, 640*480*sizeof(float));
        return img;
    };


    rgb_classification_dataset::rgb_classification_dataset(const std::string& filename, int pattern_size, int n_crops)
        :m_n_crops(n_crops),
         m_pattern_size(pattern_size){
            std::ifstream ifs(filename.c_str());
            ifs >> m_n_classes;
            ifs.get(); // consume '\n'
            for(unsigned int klass = 0; klass < (unsigned int)m_n_classes; klass++){
                std::string line;
                std::getline(ifs, line);
                if(ifs)
                    m_class_names.push_back(line);
            }
            cuvAssert(m_class_names.size() == (unsigned int)m_n_classes);
            while(ifs){
                meta_t m;
                ifs >> m.rgb_filename;
                ifs >> m.klass; // dummy value for historical reasons
                ifs >> m.klass;
                if(ifs)
                    m_meta.push_back(m);
            }
            cuvAssert(m_meta.size() > 0);
            LOG4CXX_WARN(g_log, "read `"<< filename<<"', n_classes: "<<m_n_classes<<", size: "<<m_meta.size());
            m_predictions.resize(size(), -1);
        }

    /// generates n_crops many regions of a given size within the boundaries of a given image img
    std::vector<cv::Rect> random_regions(const cv::Mat& img, int n_crops, int size){
        std::vector<cv::Rect> regions(n_crops);
        float max_x = img.cols - size;
        float max_y = img.rows - size;
        for(int i = 0; i<n_crops; i++){
           regions[i] = cv::Rect(cv::Point2f(max_x * drand48(), max_y * drand48()), cv::Size(size,size)); 
        }
        return regions;
    }

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
            cuv::convert(view, wrapped); // copies memory
        }
    }

    boost::shared_ptr<rgb_classification_dataset::patternset_t> rgb_classification_dataset::preprocess(size_t idx, boost::shared_ptr<input_t> in) {
        meta_t& meta = m_meta[idx];
        in->ID = idx;

        auto patternset = boost::make_shared<patternset_t>();

        auto regions = random_regions(in->rgb, m_n_crops, m_pattern_size);
        for (auto r : regions){
            auto pattern = boost::make_shared<pattern_t>();
            pattern->original = in;
            pattern->ground_truth_class = meta.klass;
        
            cv::Mat region = in->rgb(r);
            std::vector<cv::Mat> chans;
            cv::split(region, chans);
            pattern->rgb.resize(cuv::extents[3][m_pattern_size][m_pattern_size]); 
            dstack_mat2tens(pattern->rgb, chans); // TODO RGB/BGR conversion here
        
            patternset->push(pattern);
        }

        return patternset;
    }

    void rgb_classification_dataset::notify_done(boost::shared_ptr<pattern_t> pat) {
        boost::shared_ptr<patternset_t> set = pat->set; // NOTE: need to do this BEFORE notify_done!

        base_t::notify_done(pat);

        if(set->todo_size() == 0 && set->processing_size() == 0){
            cuv::tensor<float, cuv::host_memory_space> pred;
            for(const auto& p : set->m_done){
                if(pred.ndim() == 0)
                    pred = p->predicted_class;
                else
                    pred += p->predicted_class;
            }
            //pred /= (float)set->m_done.size();
            m_predictions[pat->original->ID] = cuv::arg_max(pred);
        }
    }

}
