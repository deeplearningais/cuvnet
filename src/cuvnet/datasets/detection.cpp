#include <fstream>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/matwrite.hpp>
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
    template<>
    boost::shared_ptr<meta_data<rgbd_detection_tag>::input_t>
    load_image<rgbd_detection_tag>(const meta_data<rgbd_detection_tag>& meta){
        auto img = boost::make_shared<rgbd_image>();
        {    
            meta_data<rgb_classification_tag> tmp_meta;
           tmp_meta.rgb_filename = meta.rgb_filename;
           //  delegate to rgb_classification
           auto tmp = load_image(tmp_meta);
           img->rgb = tmp->rgb;
           img->ID = tmp->ID;
        }
       
        img->depth = cv::imread(meta.depth_filename, CV_LOAD_IMAGE_ANYDEPTH);
        img->depth.convertTo(img->depth, CV_32F);
       
        return img;
    }

    rgb_detection_dataset::rgb_detection_dataset(const std::string& filename, int pattern_size, int n_crops)
    : m_n_crops(n_crops)
    , m_pattern_size(pattern_size)
    , m_filename(filename)
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
                ifs >> bb.rect.x;
                ifs >> bb.rect.w;   bb.rect.w -= bb.rect.x;
                ifs >> bb.rect.y;
                ifs >> bb.rect.h;   bb.rect.h -= bb.rect.y;
                m.bboxes.push_back(bb);
            }
            if(ifs)
                m_meta.push_back(m);
        }
        cuvAssert(m_n_classes >= mapcnt.size());
        cuvAssert(m_meta.size() > 0);
        LOG4CXX_WARN(g_log, "read `"<< filename<<"', n_classes: "<<m_n_classes<<", size: "<<m_meta.size());
        shuffle(false);
    }

    void rgb_detection_dataset::set_imagenet_mean(std::string filename){
            LOG4CXX_WARN(g_log, "read imagnet mean from `"<< filename<<"'");
            m_imagenet_mean = cuvnet::fromfile<float>(filename);
    }

    void rgb_detection_dataset::set_image_basepath(std::string path){
        for(auto& m : m_meta){
            m.rgb_filename = path + "/" + m.rgb_filename;
        }
    }

    void rgb_detection_dataset::notify_done(boost::shared_ptr<pattern_t> pat){
        boost::shared_ptr<patternset_t> set = pat->set; // NOTE: need to do this BEFORE notify_done!

        base_t::notify_done(pat);

        if(set->todo_size() == 0 && set->processing_size() == 0){
            std::vector<bbox> pred;
            std::vector<float> scale_org;
            int count = 0;
            for(const auto& p : set->m_done){
                float scale_x = p->region_in_original.h / m_pattern_size;
                float scale_y = p->region_in_original.w / m_pattern_size;
                
                cv::Rect margins;
                cv::RotatedRect pos_in_enlarged;
                boost::tie(margins, pos_in_enlarged) = required_padding(p->original->rgb, p->region_in_original);

                for(auto b : p->predicted_bboxes) {
                    scale_org.push_back(p->region_in_original.w / p->original->rgb.cols);
                
                    // Instructions inverse to preprocessing to move and scale 
                    // bboxes back into the perspective of the original image.
                    // Does not support rotation!

                    // [0...1] --> [0..region size]
                    b.rect.x *= m_pattern_size * scale_x; 
                    b.rect.y *= m_pattern_size * scale_y; 
                    b.rect.h *= m_pattern_size * scale_x; 
                    b.rect.w *= m_pattern_size * scale_y; 
                        
                    // determine upper left corner
                    // (x,y) was center --> upper left coordinate of bbox
                    b.rect.x -= b.rect.w/2;
                    b.rect.y -= b.rect.h/2;
                    if (p->flipped){
                        //b.rect.x = (m_pattern_size * scale_x - 1.f) - b.rect.x - b.rect.w; // falsch? aber gerade irrelevant
                        b.rect.x -= ((m_pattern_size * scale_x - 1.f) - b.rect.w); // falsch? aber gerade irrelevant
                        b.rect.x *= -1.;
                    }

                    // coordinate system origin is center of patch 
                    b.rect.x -= pos_in_enlarged.size.width/2;
                    b.rect.y -= pos_in_enlarged.size.height/2;
                    
                    // coordinate system origin is center of patch in full enlarged image
                    b.rect.x += pos_in_enlarged.center.x;
                    b.rect.y += pos_in_enlarged.center.y;
                    
                    // coordinate system origin is center of patch in full image
                    b.rect.x -= margins.x;
                    b.rect.y -= margins.y;


                    pred.push_back(b);
                    count++;
                }
            }
                
            int idx = 0;
            static std::ofstream out_f("predicted_bboxes.txt", std::ios::out);
            out_f << m_meta[pat->original->ID].rgb_filename << " ";
            out_f << count << " ";
            for (auto p : pred) {
                out_f << p.klass << " "
                      << scale_org[idx++] << " "
                      << p.rect.x << " "
                      << p.rect.y << " "
                      << p.rect.h << " "
                      << p.rect.w << " "
                      << p.confidence << " ";
                }
            out_f << std::endl;
                
            set->m_done.clear(); // prevent circles

            static int cnt = 0;
            if(++cnt % 50 == 0){
            }
            pat->set.reset();
        }
    
    }

    boost::shared_ptr<rgb_detection_dataset::patternset_t>
        rgb_detection_dataset::preprocess(size_t idx, boost::shared_ptr<input_t> in) const {
            const meta_t& meta = m_meta[idx];
            in->ID = idx;

            auto patternset = boost::make_shared<patternset_t>();

            auto regions = random_regions(in->rgb, m_n_crops, 0.25);
            //auto regions = random_regions_from_depth(in->rgb, m_n_crops, 300., 1.);
            //auto regions = covering_regions_from_depth(in->rgb, 300., 1., 1.0);

            for (auto r : regions){
                r.angle = 20 * drand48() - 10;
                auto pattern = boost::make_shared<pattern_t>();
                pattern->original = in;
                pattern->region_in_original = r;
                pattern->flipped = drand48() > 0.5f;

                cv::Mat region = extract_region(in->rgb, r, pattern->flipped, cv::INTER_LINEAR);
                float scale_x = region.cols / (float)m_pattern_size; 
                float scale_y = region.rows / (float)m_pattern_size;
                if(region.cols != m_pattern_size || region.rows != m_pattern_size)
                    cv::resize(region, region, cv::Size(m_pattern_size, m_pattern_size), 0., 0., cv::INTER_LINEAR);
                {
                    // now translate the bounding boxes
                    cv::Rect margins;
                    cv::RotatedRect pos_in_enlarged;
                    boost::tie(margins, pos_in_enlarged) = required_padding(in->rgb, r);
                    for(const auto& bb : meta.bboxes){
                        cv::RotatedRect tmp(
                                cv::Point2f(
                                    bb.rect.x + bb.rect.w/2.f,
                                    bb.rect.y + bb.rect.h/2.f),
                                cv::Size(bb.rect.w, bb.rect.h), 0);

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
                            bounds_rot[i].x = (bounds[i].x * cos(deg2Rad(-r.angle)) - bounds[i].y*sin(deg2Rad(-r.angle)));
                            bounds_rot[i].y = (bounds[i].x * sin(deg2Rad(-r.angle)) + bounds[i].y*cos(deg2Rad(-r.angle)));
                        }
                        cv::Rect r = cv::boundingRect(bounds_rot);

                        bbox pbb;
                        // origin patch center  -> upper left corner
                        pbb.rect.x = r.x + pos_in_enlarged.size.width/2;
                        pbb.rect.y = r.y + pos_in_enlarged.size.height/2;
                        pbb.rect.w = r.width;
                        pbb.rect.h = r.height;
                        if(pattern->flipped){
                            pbb.rect.x = (m_pattern_size * scale_x - 1.f) - pbb.rect.x - pbb.rect.w;
                        }

                        pbb.rect.x += pbb.rect.w/2;
                        pbb.rect.y += pbb.rect.h/2;

                        // scale bboxes to relative values ([0..1])
                        pbb.rect.x /= m_pattern_size * scale_x; 
                        pbb.rect.y /= m_pattern_size * scale_y; 
                        pbb.rect.h /= m_pattern_size * scale_x; 
                        pbb.rect.w /= m_pattern_size * scale_y; 

                        // determine points of greatest possible rectangle within patch and bbox
                        float ulx = std::max(0.f, std::min(1.f, (float) (pbb.rect.x - pbb.rect.w/2)));
                        float uly = std::max(0.f, std::min(1.f, (float) (pbb.rect.y - pbb.rect.h/2)));
                        float lrx = std::max(0.f, std::min(1.f, (float) (pbb.rect.x + pbb.rect.w/2)));
                        float lry = std::max(0.f, std::min(1.f, (float) (pbb.rect.y + pbb.rect.h/2)));

                        // determine ratio of visible area of bbox within the patch
                        float full_area = pbb.rect.w * pbb.rect.h;
                        float in_area = (lrx - ulx) * (lry - uly);
                        // only include boxes of whose area is 50% visible
                        if ((in_area / full_area) < 0.5)
                            continue;
                        // include only bboxes whose area is less than 20% of the total pattern size
                        if ((in_area / (1*1)) < 0.2)
                            continue;

                        pbb.klass = bb.klass;
                        pbb.truncated = bb.truncated;
                        pattern->bboxes.push_back(pbb);
                    }
                }

                std::vector<cv::Mat> chans;
                cv::split(region, chans);
                pattern->rgb.resize(cuv::extents[3][m_pattern_size][m_pattern_size]); 
                
                dstack_mat2tens(pattern->rgb, chans);

                if(m_imagenet_mean.ptr())
                    pattern->rgb -= m_imagenet_mean;

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
            cv::rectangle(cvrgb, cv::Point(bb.rect.x, bb.rect.y), cv::Point(bb.rect.x+bb.rect.w, bb.rect.y+bb.rect.h), cv::Scalar(1));
        }
        cv::imshow(name, cvrgb);
    }

    void rgb_detection_dataset::aggregate_statistics(const unsigned int n){
        std::ofstream bboxes_f;
      
        // remove trailing .txt 
        std::string base = m_filename.substr(0, m_filename.length() - 4); 
        bboxes_f.open(base + "_bboxes.txt");

        std::vector<cuv::tensor<float, cuv::host_memory_space> > v;
        cuv::tensor<float, cuv::host_memory_space> avg;
        int N = 0;
        for (unsigned int i = 0; i < n; i++) {
            auto pattern_set = next(i); 

            bool start_seq = i % 1000 == 0;
            if(start_seq){
                if(i>0){
                    v.push_back(avg / (float) N);
                    N = 0;
                }
                auto pat = pattern_set->get_for_processing();
                avg = pat->rgb;
            }

            while(! pattern_set->m_todo.empty()) {
                auto pat = pattern_set->get_for_processing();

                avg += pat->rgb;
                N ++;

                for (auto bb : pat->bboxes) {
                    bboxes_f 
                        << bb.klass << " "
                        << bb.rect.x << " " 
                        << bb.rect.y << " " 
                        << bb.rect.h << " " 
                        << bb.rect.w << std::endl; 
                }
            }
        }
        if(N > 0)
            v.push_back(avg / (float)N);
        avg = 0.f;
        for(auto& a : v)
            avg += a;
        avg /= (float)v.size();

        if(!m_imagenet_mean.ptr()) {
            cuvnet::tofile(base + "_mean.npy", avg);
        } else {
            std::string l = "zero mean deviation: " + std::to_string(cuv::mean(avg)) + " ";
            LOG4CXX_WARN(g_log, l);
        }
        
        bboxes_f.close();
    };



/* RGB D DETECTION */



    rgbd_detection_dataset::rgbd_detection_dataset(const std::string& filename, int pattern_size, int n_crops)
    : m_n_crops(n_crops)
    , m_pattern_size(pattern_size)
    , m_filename(filename)
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
            ifs >> m.depth_filename;
            ifs >> n_bboxes;
            for (int i = 0; i < n_bboxes; ++i) {
                bbox bb;
                ifs >> bb.klass;
                mapcnt[bb.klass]++;
                //ifs >> bb.truncated;
                ifs >> bb.rect.x;
                ifs >> bb.rect.w;   bb.rect.w -= bb.rect.x;
                ifs >> bb.rect.y;
                ifs >> bb.rect.h;   bb.rect.h -= bb.rect.y;
                m.bboxes.push_back(bb);
            }
            if(ifs)
                m_meta.push_back(m);
        }
        cuvAssert(m_n_classes >= mapcnt.size());
        cuvAssert(m_meta.size() > 0);
        LOG4CXX_WARN(g_log, "read `"<< filename<<"', n_classes: "<<m_n_classes<<", size: "<<m_meta.size());
        shuffle(false);
    }

    void rgbd_detection_dataset::set_imagenet_mean(std::string filename){
            LOG4CXX_WARN(g_log, "read imagnet mean from `"<< filename<<"'");
            m_imagenet_mean = cuvnet::fromfile<float>(filename);
            m_imagenet_mean_depth = cuvnet::fromfile<float>(filename.substr(0, filename.length()-4) + "_depth.npy");
    }

    void rgbd_detection_dataset::set_image_basepath(std::string path){
        for(auto& m : m_meta){
            m.rgb_filename = path + "/" + m.rgb_filename;
            m.depth_filename = path + "/" + m.depth_filename;
        }
    }

    void rgbd_detection_dataset::notify_done(boost::shared_ptr<pattern_t> pat){
        boost::shared_ptr<patternset_t> set = pat->set; // NOTE: need to do this BEFORE notify_done!

        base_t::notify_done(pat);

        if(set->todo_size() == 0 && set->processing_size() == 0){
            std::vector<bbox> pred;
            std::vector<float> scale_org;
            int count = 0;
            for(const auto& p : set->m_done){
                float scale_x = p->region_in_original.h / m_pattern_size;
                float scale_y = p->region_in_original.w / m_pattern_size;
                
                cv::Rect margins;
                cv::RotatedRect pos_in_enlarged;
                boost::tie(margins, pos_in_enlarged) = required_padding(p->original->rgb, p->region_in_original);

                for(auto b : p->predicted_bboxes) {
                    scale_org.push_back(p->region_in_original.w / p->original->rgb.cols);
                
                    // Instructions inverse to preprocessing to move and scale 
                    // bboxes back into the perspective of the original image.
                    // Does not support rotation!

                    // [0...1] --> [0..region size]
                    b.rect.x *= m_pattern_size * scale_x; 
                    b.rect.y *= m_pattern_size * scale_y; 
                    b.rect.h *= m_pattern_size * scale_x; 
                    b.rect.w *= m_pattern_size * scale_y; 
                        
                    // determine upper left corner
                    // (x,y) was center --> upper left coordinate of bbox
                    b.rect.x -= b.rect.w/2;
                    b.rect.y -= b.rect.h/2;
                    if (p->flipped){
                        b.rect.x -= ((m_pattern_size * scale_x - 1.f) - b.rect.w);
                        b.rect.x *= -1.;
                    }

                    // coordinate system origin is center of patch 
                    b.rect.x -= pos_in_enlarged.size.width/2;
                    b.rect.y -= pos_in_enlarged.size.height/2;
                    
                    // coordinate system origin is center of patch in full enlarged image
                    b.rect.x += pos_in_enlarged.center.x;
                    b.rect.y += pos_in_enlarged.center.y;
                    
                    // coordinate system origin is center of patch in full image
                    b.rect.x -= margins.x;
                    b.rect.y -= margins.y;


                    pred.push_back(b);
                    count++;
                }
            }
                
            int idx = 0;
            static std::ofstream out_f("predicted_bboxes.txt", std::ios::out);
            out_f << m_meta[pat->original->ID].rgb_filename << " ";
            //out_f << m_meta[pat->original->ID].depth_filename << " ";
            out_f << count << " ";
            for (auto p : pred) {
                out_f << p.klass << " "
                      << scale_org[idx++] << " "
                      << p.rect.x << " "
                      << p.rect.y << " "
                      << p.rect.h << " "
                      << p.rect.w << " "
                      << p.confidence << " ";
                }
            out_f << std::endl;
                
            set->m_done.clear(); // prevent circles

            static int cnt = 0;
            if(++cnt % 50 == 0){
            }
            pat->set.reset();
        }
    
    }

    boost::shared_ptr<rgbd_detection_dataset::patternset_t>
        rgbd_detection_dataset::preprocess(size_t idx, boost::shared_ptr<input_t> in) const {
            const meta_t& meta = m_meta[idx];
            in->ID = idx;

            auto patternset = boost::make_shared<patternset_t>();

            //auto regions = random_regions(in->rgb, m_n_crops, 0.25);
            auto regions = random_regions_from_depth(in->depth, m_n_crops, 300., 1.);

            for (auto r : regions){
                r.angle = 20 * drand48() - 10;
                auto pattern = boost::make_shared<pattern_t>();
                pattern->original = in;
                pattern->region_in_original = r;
                pattern->flipped = drand48() > 0.5f;

                cv::Mat region = extract_region(in->rgb, r, pattern->flipped, cv::INTER_LINEAR);
                cv::Mat region_depth = extract_region(in->depth, r, pattern->flipped, cv::INTER_LINEAR);
                float scale_x = region.cols / (float)m_pattern_size; 
                float scale_y = region.rows / (float)m_pattern_size;
                if(region.cols != m_pattern_size || region.rows != m_pattern_size) {
                    cv::resize(region, region, cv::Size(m_pattern_size, m_pattern_size), 0., 0., cv::INTER_LINEAR);
                    cv::resize(region_depth, region_depth, cv::Size(m_pattern_size, m_pattern_size), 0., 0., cv::INTER_LINEAR);
                }
                {
                    // now translate the bounding boxes
                    cv::Rect margins;
                    cv::RotatedRect pos_in_enlarged;
                    boost::tie(margins, pos_in_enlarged) = required_padding(in->rgb, r);
                    for(const auto& bb : meta.bboxes){
                        cv::RotatedRect tmp(
                                cv::Point2f(
                                    bb.rect.x + bb.rect.w/2.f,
                                    bb.rect.y + bb.rect.h/2.f),
                                cv::Size(bb.rect.w, bb.rect.h), 0);

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
                            bounds_rot[i].x = (bounds[i].x * cos(deg2Rad(-r.angle)) - bounds[i].y*sin(deg2Rad(-r.angle)));
                            bounds_rot[i].y = (bounds[i].x * sin(deg2Rad(-r.angle)) + bounds[i].y*cos(deg2Rad(-r.angle)));
                        }
                        cv::Rect r = cv::boundingRect(bounds_rot);

                        bbox pbb;
                        // origin patch center  -> upper left corner
                        pbb.rect.x = r.x + pos_in_enlarged.size.width/2;
                        pbb.rect.y = r.y + pos_in_enlarged.size.height/2;
                        pbb.rect.w = r.width;
                        pbb.rect.h = r.height;
                        if(pattern->flipped){
                            pbb.rect.x = (m_pattern_size * scale_x - 1.f) - pbb.rect.x - pbb.rect.w;
                        }

                        pbb.rect.x += pbb.rect.w/2;
                        pbb.rect.y += pbb.rect.h/2;

                        // scale bboxes to relative values ([0..1])
                        pbb.rect.x /= m_pattern_size * scale_x; 
                        pbb.rect.y /= m_pattern_size * scale_y; 
                        pbb.rect.h /= m_pattern_size * scale_x; 
                        pbb.rect.w /= m_pattern_size * scale_y; 

                        // determine points of greatest possible rectangle within patch and bbox
                        float ulx = std::max(0.f, std::min(1.f, (float) (pbb.rect.x - pbb.rect.w/2)));
                        float uly = std::max(0.f, std::min(1.f, (float) (pbb.rect.y - pbb.rect.h/2)));
                        float lrx = std::max(0.f, std::min(1.f, (float) (pbb.rect.x + pbb.rect.w/2)));
                        float lry = std::max(0.f, std::min(1.f, (float) (pbb.rect.y + pbb.rect.h/2)));

                        // determine ratio of visible area of bbox within the patch
                        float full_area = pbb.rect.w * pbb.rect.h;
                        float in_area = (lrx - ulx) * (lry - uly);
                        // only include boxes of whose area is 50% visible
                        if ((in_area / full_area) < 0.5)
                            continue;
                        //// include only bboxes whose area is less than 20% of the total pattern size
                        //if ((in_area / (1*1)) < 0.2)
                        //    continue;

                        pbb.klass = bb.klass;
                        pbb.truncated = bb.truncated;
                        pattern->bboxes.push_back(pbb);
                    }
                }

                std::vector<cv::Mat> chans;
                cv::split(region, chans);
                pattern->rgb.resize(cuv::extents[3][m_pattern_size][m_pattern_size]); 
                dstack_mat2tens(pattern->rgb, chans);

                cv::split(region_depth, chans);
                pattern->depth.resize(cuv::extents[1][m_pattern_size][m_pattern_size]); 
                dstack_mat2tens(pattern->depth, chans);
                
                if(m_imagenet_mean.ptr()) {
                    pattern->rgb -= m_imagenet_mean;
                    pattern->depth -= m_imagenet_mean_depth;
                }

                patternset->push(pattern);
            }

            return patternset;
    }

    void meta_data<rgbd_detection_tag>::show(std::string name, const pattern_t& pat){
        std::cout << "showing depth currently not supported" << std::endl;
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
            cv::rectangle(cvrgb, cv::Point(bb.rect.x, bb.rect.y), cv::Point(bb.rect.x+bb.rect.w, bb.rect.y+bb.rect.h), cv::Scalar(1));
        }
        cv::imshow(name, cvrgb);
    }

    void rgbd_detection_dataset::aggregate_statistics(const unsigned int n_batches){
        std::ofstream bboxes_f;

        // remove trailing .txt 
        std::string base = m_filename.substr(0, m_filename.length() - 4); 
        bboxes_f.open(base + "_bboxes.txt");

        std::vector<cuv::tensor<float, cuv::host_memory_space> > v;
        std::vector<cuv::tensor<float, cuv::host_memory_space> > vd;
        cuv::tensor<float, cuv::host_memory_space> avg;
        cuv::tensor<float, cuv::host_memory_space> avg_depth;
        int N = 0;
        for (unsigned int i = 0; i < n_batches; i++) {
            auto pattern_set = next(i); 

            bool start_seq = i % 1000 == 0;
            if(start_seq){
                if(i>0){
                    v.push_back(avg / (float) N);
                    N = 0;
                    std::cout << ".";
                }
                auto pat = pattern_set->get_for_processing();
                avg = pat->rgb;
                avg_depth = pat->depth;
            }

            while(! pattern_set->m_todo.empty()) {
                auto pat = pattern_set->get_for_processing();

                avg += pat->rgb;
                avg_depth += pat->depth;
                N ++;

                for (auto bb : pat->bboxes) {
                    bboxes_f 
                        << bb.klass << " "
                        << bb.rect.x << " " 
                        << bb.rect.y << " " 
                        << bb.rect.h << " " 
                        << bb.rect.w << std::endl; 
                }
            }
        }
        if(N > 0) {
            v.push_back(avg / (float)N);
            vd.push_back(avg_depth / (float)N);
        }
        avg = 0.f;
        for(auto& a : v)
            avg += a;
        avg /= (float)v.size();
        
        avg_depth = 0.f;
        for(auto& ad : vd)
            avg_depth += ad;
        avg_depth /= (float)v.size();

        if(!m_imagenet_mean.ptr()) {
            cuvnet::tofile(base + "_mean.npy", avg);
            cuvnet::tofile(base + "_mean_depth.npy", avg_depth);
        } else {
            std::string l = "zero mean deviation: " + std::to_string((cuv::mean(avg)+cuv::mean(avg_depth)/2.f)) + " ";
            LOG4CXX_WARN(g_log, l);
        }

        bboxes_f.close();

    };

}
