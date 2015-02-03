#include <fstream>
#include <cuvnet/tools/logging.hpp>
#include <boost/tuple/tuple.hpp>
#include "image_types.hpp"
#include "detection.hpp"

namespace
{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("detection"));
}


namespace datasets
{
    boost::shared_ptr<meta_data<rgb_detection_tag>::input_t> load_image(const meta_data<rgb_detection_tag>& meta){
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
        m_n_classes = mapcnt.size();
        cuvAssert(m_meta.size() > 0);
        LOG4CXX_WARN(g_log, "read `"<< filename<<"', n_classes: "<<m_n_classes<<", size: "<<m_meta.size());
        shuffle();
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
                auto pattern = boost::make_shared<pattern_t>();
                pattern->original = in;
                pattern->region_in_original = r;
                pattern->flipped = drand48() > 0.5f;

                cv::Mat region = extract_region(in->rgb, r, pattern->flipped, cv::INTER_LINEAR);
                {
                    // now translate the bounding boxes
                    cv::Rect margins;
                    cv::RotatedRect pos_in_enlarged;
                    boost::tie(margins,pos_in_enlarged) = required_padding(in->rgb, r);
                    for(const auto& bb : meta.bboxes){
                        cv::RotatedRect tmp(
                                cv::Point2f(
                                    bb.x0+ bb.w/2.f,
                                    bb.y0+bb.h/2.f),
                                cv::Size(bb.w, bb.h), 0);

                        // add left and top margins
                        tmp.center.x += margins.x;
                        tmp.center.y += margins.y;

                        // put in coordinates of pos_in_enlarged
                        tmp.center -= pos_in_enlarged.center;

                        // rotate in reference frame of 
                        tmp.angle += pos_in_enlarged.angle;

                        cv::Rect r = tmp.boundingRect();
                        bbox pbb;
                        pbb.x0 = r.tl().x;
                        pbb.y0 = r.tl().y;
                        pbb.w  = r.size().width;
                        pbb.h  = r.size().height;
                        pattern->bboxes.push_back(pbb);
                    }
                }
                // TODO support scaling

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
}
