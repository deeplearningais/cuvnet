#include <fstream>
#include <cuvnet/tools/logging.hpp>
#include <boost/tuple/tuple.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "image_types.hpp"
#include "detection.hpp"

namespace
{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("detection"));

    //static double rad2Deg(double rad){return rad*(180/M_PI);}//Convert radians to degrees
    static double deg2Rad(double deg){return deg*(M_PI/180);}//Convert degrees to radians
}


namespace datasets
{
    template<>
    boost::shared_ptr<meta_data<rgb_detection_tag>::input_t>
        load_image<rgb_detection_tag>(const meta_data<rgb_detection_tag>& meta){
        meta_data<rgb_classification_tag> tmp_meta;
        tmp_meta.rgb_filename = meta.rgb_filename;
        // delegate to rgb_classification
        return load_image(tmp_meta);
    }

    rgb_detection_dataset::rgb_detection_dataset(const std::string& filename, int pattern_size, int n_crops)
    : m_n_crops(n_crops)
    , m_pattern_size(pattern_size)
    {
        std::ifstream ifs(filename.c_str());
        cuvAssert(ifs.is_open() && ifs.good());
        ifs >> m_n_classes;
        ifs.get(); // consume '\n'
        for(unsigned int klass = 0; klass < m_n_classes; klass++){
            std::string line;
            std::getline(ifs, line);
            if(ifs)
                m_class_names.push_back(line);
        }
        cuvAssert(m_class_names.size() == m_n_classes);

        std::map<int,int> mapcnt;
        while(ifs){
            int n_bboxes;
            meta_t m;
            ifs >> m.rgb_filename;
            ifs >> n_bboxes;
            for (int i = 0; i < n_bboxes; ++i) {
                bbox bb;
                ifs >> bb.klass;
                mapcnt[bb.klass]++;
                ifs >> bb.truncated;
                ifs >> bb.x0;
                ifs >> bb.w;   bb.w -= bb.x0;
                ifs >> bb.y0;
                ifs >> bb.h;   bb.h -= bb.y0;
                m.bboxes.push_back(bb);
            }
            if(ifs)
                m_meta.push_back(m);
        }
        cuvAssert(m_n_classes == mapcnt.size());
        cuvAssert(m_meta.size() > 0);
        LOG4CXX_WARN(g_log, "read `"<< filename<<"', n_classes: "<<m_n_classes<<", size: "<<m_meta.size());
        shuffle(false);
    }

    void rgb_detection_dataset::set_imagenet_mean(std::string filename){
            LOG4CXX_WARN(g_log, "read imagnet mean from `"<< filename<<"'");
            std::ifstream meanifs(filename.c_str());
            cuvAssert(meanifs.is_open());
            cuvAssert(meanifs.good());
            m_imagenet_mean.resize(cuv::extents[3][m_pattern_size][m_pattern_size]);
            meanifs.read((char*)m_imagenet_mean.ptr(), m_imagenet_mean.memsize());
    }

    void rgb_detection_dataset::set_image_basepath(std::string path){
        for(auto& m : m_meta){
            m.rgb_filename = path + "/" + m.rgb_filename;
        }
    }
    void rgb_detection_dataset::notify_done(boost::shared_ptr<pattern_t> pat){
        // TODO to be written
    }
    boost::shared_ptr<rgb_detection_dataset::patternset_t>
        rgb_detection_dataset::preprocess(size_t idx, boost::shared_ptr<input_t> in) const {
            const meta_t& meta = m_meta[idx];
            in->ID = idx;

            auto patternset = boost::make_shared<patternset_t>();

            auto regions = random_regions(in->rgb, m_n_crops, m_pattern_size);
            for (auto r : regions){
                r.center.x += 90;
                r.center.y += 120;
                //r.size.width = 1.5 * in->rgb.cols;
                //r.size.height = 1.5 * in->rgb.rows;
                r.angle = 25;
                auto pattern = boost::make_shared<pattern_t>();
                pattern->original = in;
                pattern->region_in_original = r;
                pattern->flipped = drand48() > 0.5f;

                cv::Mat region = extract_region(in->rgb, r, pattern->flipped, cv::INTER_LINEAR);
                {
                    // now translate the bounding boxes
                    cv::Rect margins;
                    cv::RotatedRect pos_in_enlarged;
                    boost::tie(margins, pos_in_enlarged) = required_padding(in->rgb, r);
                    for(const auto& bb : meta.bboxes){
                        cv::RotatedRect tmp(
                                cv::Point2f(
                                    bb.x0 + bb.w/2.f,
                                    bb.y0 + bb.h/2.f),
                                cv::Size(bb.w, bb.h), 0);

                        // 1. add left and top margins
                        tmp.center.x += margins.x;
                        tmp.center.y += margins.y;

                        // 2. put in coordinates of pos_in_enlarged
                        tmp.center -= pos_in_enlarged.center;

                        std::vector<cv::Point2f> bounds(4);
                        tmp.points(&bounds[0]);

                        // now rotate these points around the "origin", i.e. the center of pos_in_enlarged
                        std::vector<cv::Point2f> bounds_rot(4);
                        for (int i = 0; i < 4; ++i) {
                            bounds_rot[i].x = bounds[i].x * cos(deg2Rad(-r.angle)) - bounds[i].y*sin(deg2Rad(-r.angle));
                            bounds_rot[i].y = bounds[i].x * sin(deg2Rad(-r.angle)) + bounds[i].y*cos(deg2Rad(-r.angle));
                        }
                        cv::Rect r = cv::boundingRect(bounds_rot);

                        bbox pbb;
                        pbb.x0 = r.x + pos_in_enlarged.size.width/2;
                        pbb.y0 = r.y + pos_in_enlarged.size.height/2;
                        pbb.w  = r.width;
                        pbb.h  = r.height;
                        pattern->bboxes.push_back(pbb);
                    }
                }
                // TODO support scaling, flip bounding boxes

                std::vector<cv::Mat> chans;
                cv::split(region, chans);
                pattern->rgb.resize(cuv::extents[3][m_pattern_size][m_pattern_size]); 
                
                dstack_mat2tens(pattern->rgb, chans);

                if(m_imagenet_mean.ptr())
                    pattern->rgb -= m_imagenet_mean;
                else
                    pattern->rgb -= 121.f;  // poor man's approximation here...

                patternset->push(pattern);
            }

            return patternset;
    }

    void meta_data<rgb_detection_tag>::show(std::string name, const pattern_t& pat){
        auto rgb = pat.rgb.copy();
        rgb -= cuv::minimum(rgb);
        rgb /= cuv::maximum(rgb);
        std::vector<cv::Mat> channels(3, cv::Mat(rgb.shape(1), rgb.shape(2), CV_32FC1));
        for (int i = 0; i < 3; ++i) {
            memcpy((char*)channels[i].ptr<float>(), rgb[cuv::indices[i]].ptr(), sizeof(float) * rgb.shape(1) * rgb.shape(2));
        }
        cv::Mat cvrgb;
        cv::merge(channels, cvrgb);
        for (const auto& bb : pat.bboxes) {
            cv::rectangle(cvrgb, cv::Point(bb.x0, bb.y0), cv::Point(bb.x0+bb.w, bb.y0+bb.h), cv::Scalar(1));
        }
        cv::imshow(name, cvrgb);
    }
}
