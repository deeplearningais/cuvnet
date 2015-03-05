#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <third_party/cnpy/cnpy.h>
#include <cuvnet/tools/logging.hpp>
#include <fstream>

#include "image_types.hpp"
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
        :m_n_crops(n_crops)
        ,m_pattern_size(pattern_size)
        ,m_filename(filename){
            std::ifstream ifs(filename.c_str());
            unsigned int n_cls;
            ifs >> n_cls;
            ifs.get(); // consume '\n'
            for(unsigned int klass = 0; klass < n_cls; klass++){
                std::string line;
                std::getline(ifs, line);
                if(ifs)
                    m_class_names.push_back(line);
            }
            cuvAssert(m_class_names.size() == n_cls);
            while(ifs){
                meta_t m;
                ifs >> m.rgb_filename;
                //ifs >> m.klass; // dummy value for historical reasons
                ifs >> m.klass;
                if(ifs)
                    m_meta.push_back(m);
            }
            cuvAssert(m_meta.size() > 0);
            LOG4CXX_WARN(g_log, "read `"<< filename<<"', n_classes: "<<n_cls<<", size: "<<m_meta.size());
            m_predictions.resize(this->size());
            shuffle(false);
        }

    void rgb_classification_dataset::clear_predictions(){
        m_predictions.clear();
        m_predictions.resize(this->size());
    }

    void rgb_classification_dataset::set_input_map_mean(std::string filename){
            LOG4CXX_WARN(g_log, "read imagnet mean from `"<< filename<<"'");
            std::ifstream meanifs(filename.c_str());
            cuvAssert(meanifs.is_open());
            cuvAssert(meanifs.good());
            m_input_map_mean.resize(cuv::extents[3][m_pattern_size][m_pattern_size]);
            meanifs.read((char*)m_input_map_mean.ptr(), m_input_map_mean.memsize());
    }


    cv::RotatedRect
    region_from_depth(const cv::Mat& depth, cv::Point2f center, double scale_fact, double min_dist){
        double d = depth.at<float>(center) / 1000.;  // convert to meters
        double f;
        if(d!=d)
            f = min_dist;
        else
            f = scale_fact / std::max(min_dist, d);
        //std::cout << "dist: " << d<< " f: " << f <<std::endl;
#ifdef FIXED_SIZE
        f = 135;
#endif
        f = std::max(15., f);
        cv::Size size(f, f);

        return cv::RotatedRect(center, size, 0.);
    }

    /// generates n_crops many regions of a given size within the boundaries of a given image img
    std::vector<cv::RotatedRect> random_regions(const cv::Mat& img, int n_crops, float min_size_frac){
        std::vector<cv::RotatedRect> regions(n_crops);
        for(int i = 0; i<n_crops; i++){
            float f = std::exp(drand48() * (std::log(1.f) - std::log(min_size_frac)) + std::log(min_size_frac));
            float size = std::max(img.cols, img.rows) * f;
            float max_x = img.cols - size;
            float max_y = img.rows - size;
            cv::Rect r(cv::Point2f(max_x * drand48(), max_y * drand48()), cv::Size(size,size)); 
            regions[i] = cv::RotatedRect(cv::Point2f(r.x+size/2, r.y+size/2), cv::Size(size,size), 0.f);
        }
        return regions;
    }

    /// generates n_crops many regions within the boundaries of a given image img, scaling wrt to depth of center 
    std::vector<cv::RotatedRect> random_regions_from_depth(const cv::Mat& depth, int n_crops, double scale_fact, double min_dist){
        std::vector<cv::RotatedRect> regions;
        for (int i=0; i<n_crops; i++){
            int margin = std::min(depth.rows, depth.cols)/10;
            cv::Point2f center(
                margin + drand48()*(depth.cols-2*margin),
                margin + drand48()*(depth.rows-2*margin));
            regions.push_back(region_from_depth(depth, center, scale_fact, min_dist));
        }
        return regions;
    }

    template<class T>
    void find_zeros(const cv::Mat_<T>& binary, std::vector<cv::Point> &idx) {
        assert(binary.cols > 0 && binary.rows > 0 && binary.channels() == 1 && binary.channels() == 1);
        const int M = binary.rows;
        const int N = binary.cols;
        for (int m = 0; m < M; ++m) {
            const T* bin_ptr = (T*) binary.ptr(m);
            for (int n = 0; n < N; ++n) {
                if (bin_ptr[n] == 0) idx.push_back(cv::Point(n,m));
            }
        }
    }

    std::vector<cv::RotatedRect>
        covering_regions_from_depth(const cv::Mat& depth, double scale_fact, double min_dist, double output_fraction){
            cv::vector<cv::RotatedRect> rects;
            cv::Mat_<int> counts = cv::Mat_<int>::zeros(depth.rows, depth.cols);
            //cv::Mat_<int> intimg(depth.rows, depth.cols);
            std::vector<cv::Point> pt;
            std::vector<double> weights;
            while(true){
                pt.clear();
                find_zeros(counts, pt);
                if(pt.empty())
                    break;
                weights.resize(pt.size());
                double thresh = 0.;
                {
                    double sum = 0.;
                    std::vector<double>::iterator it = weights.begin();
                    for(cv::Point& p : pt){
                        double d = depth.at<float>(p) / 1000.;  // meters
#ifdef FIXED_SIZE
                        d = 1.;
#endif
                        if(d!=d || d > 10. || d == 0)
                            *it++ = 0.;
                        else{
                            *it++ = d;
                            sum += d;
                        }
                    }
                    thresh = drand48() * sum;
                }
                int idx = 0;
                {
                    double sum = 0.;
                    for(unsigned int i = 0; i<weights.size(); i++){
                        sum += weights[i];
                        if(sum > thresh) {
                            idx = i;
                            break;
                        }
                    }
                }
                cv::Point loc(pt[idx]);
                auto rr = region_from_depth(depth, loc, scale_fact, min_dist);
                rects.push_back(rr);

                rr.size.width *= output_fraction;
                rr.size.height *= output_fraction;
                cv::Rect br(rr.center.x-rr.size.width/2, rr.center.y-rr.size.height/2,rr.size.width,rr.size.height);
                br = br & cv::Rect(0, 0, depth.cols, depth.rows);
                counts(br) += 1;
            }
            return rects;
        }


    /**
     * every entry in Rect tells how much to extend the corresponding image in
     * m so that the rotated rect in ir fits into the new image.
     *
     * @return the required margins and the rotated rect w.r.t. the padded image.
     */
    std::pair<cv::Rect, cv::RotatedRect>
    required_padding(const cv::Mat& m, const cv::RotatedRect& ir){
        cv::Rect br = ir.boundingRect();
        br.x -= 1;
        br.y -= 1;
        br.width  += 2; // one in every direction
        br.height += 2;
        std::pair<cv::Rect, cv::RotatedRect> ret;
        ret.first.x = std::max(0, -br.x);
        ret.first.y = std::max(0, -br.y);
        ret.first.width  = std::max(0,  br.br().x - m.cols + 1);
        ret.first.height = std::max(0,  br.br().y - m.rows + 1);
        ret.second = ir;
        ret.second.center += cv::Point2f(ret.first.x, ret.first.y);
        assert(ret.second.boundingRect().x >= 0);
        assert(ret.second.boundingRect().y >= 0);
        assert(ret.second.boundingRect().width + ret.second.boundingRect().x <= m.cols + ret.first.x + ret.first.width);
        assert(ret.second.boundingRect().height + ret.second.boundingRect().y <= m.rows + ret.first.y + ret.first.height);
        return ret;
    }
    cv::Mat extract_region(const cv::Mat& m, const cv::RotatedRect& ir, bool flipped, int interpolation, int bordertype, int value){
        cv::Mat M, enlarged, rotated, cropped;
        cv::Rect margins;
        cv::RotatedRect pos_in_enlarged;
        boost::tie(margins,pos_in_enlarged) = required_padding(m, ir);
        cv::copyMakeBorder(m, enlarged, 
                margins.y, margins.height, margins.x, margins.width, 
                bordertype, value);

        cv::Size rect_size = pos_in_enlarged.size;
        float angle = pos_in_enlarged.angle;
        if(angle == 0.){
            cv::Rect rr(
                    pos_in_enlarged.center.x - pos_in_enlarged.size.width/2,
                    pos_in_enlarged.center.y - pos_in_enlarged.size.height/2,
                    pos_in_enlarged.size.width, pos_in_enlarged.size.height);
            cropped = enlarged(rr);
        }
        else{
            if (pos_in_enlarged.angle < -45.) {
                angle += 90.0;
                std::swap(rect_size.width, rect_size.height);
            }
            M = cv::getRotationMatrix2D(pos_in_enlarged.center, angle, 1.0);
            cv::warpAffine(enlarged, rotated, M, enlarged.size(), interpolation);
            cv::getRectSubPix(rotated, rect_size, pos_in_enlarged.center, cropped);
            assert(cropped.rows == cropped.cols);
        }
        if(flipped)
            cv::flip(cropped, cropped, 1);
        if(!cropped.isContinuous())
            cropped = cropped.clone();
        return cropped;
    }

    void dstack_mat2tens(cuv::tensor<float, cuv::host_memory_space>& tens,
            const std::vector<cv::Mat>& src, bool reverse){
        assert(src.size() >= 1);
        int d = src.size();
        int h = src.front().rows;
        int w = src.front().cols;

        tens.resize(cuv::extents[d][h][w]);
        for (int i = 0; i < d; ++i)
        {
            int other = reverse ? d-1-i : i;
            if(src[0].type() == CV_8UC1){
                typedef unsigned char src_dtype;
                cuv::tensor<src_dtype, cuv::host_memory_space> wrapped(cuv::extents[h][w], (src_dtype*) src[other].data);
                cuv::tensor<float, cuv::host_memory_space> view = cuv::tensor_view<float, cuv::host_memory_space>(tens, cuv::indices[i][cuv::index_range()][cuv::index_range()]);
                cuv::convert(view, wrapped); // copies memory
            }else if(src[0].type() == CV_32FC1){
                typedef float src_dtype;
                cuv::tensor<src_dtype, cuv::host_memory_space> wrapped(cuv::extents[h][w], (src_dtype*) src[other].data);
                cuv::tensor_view<float, cuv::host_memory_space> view(tens, cuv::indices[i][cuv::index_range()][cuv::index_range()]);
                view = wrapped; // copies memory
            }
        }
    }

    boost::shared_ptr<rgb_classification_dataset::patternset_t>
    rgb_classification_dataset::preprocess(size_t idx, boost::shared_ptr<input_t> in) const{
        const meta_t& meta = m_meta[idx];
        in->ID = idx;

        auto patternset = boost::make_shared<patternset_t>();

        auto regions = random_regions(in->rgb, m_n_crops, m_pattern_size);
        for (auto r : regions){
            auto pattern = boost::make_shared<pattern_t>();
            pattern->original = in;
            pattern->ground_truth_class = meta.klass;
            pattern->region_in_original = r;
            pattern->flipped = drand48() > 0.5f;
        
            cv::Mat region = extract_region(in->rgb, r, pattern->flipped, cv::INTER_LINEAR, cv::BORDER_REFLECT_101, 0);

            std::vector<cv::Mat> chans;
            cv::split(region, chans);
            pattern->rgb.resize(cuv::extents[3][m_pattern_size][m_pattern_size]); 
            dstack_mat2tens(pattern->rgb, chans); // TODO RGB/BGR conversion here

            if(m_input_map_mean.ptr())
                pattern->rgb -= m_input_map_mean;
            else
                pattern->rgb -= 121.f;  // poor man's approximation here...
        
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
            m_predictions[pat->original->ID] = pred;

            set->m_done.clear(); // prevent circles

            static int cnt = 0;
            if(++cnt % 50 == 0){
                get_zero_one(1);
                get_zero_one(5);
            }
        }
    }

    float rgb_classification_dataset::get_zero_one(int k){
        size_t cnt = 0;  // counts for how many instances we /have/ a prediction
        size_t correct = 0;
        for (size_t i = 0; i < size(); ++i) {
            if(!m_predictions[i].ptr())
                continue;
            cnt++;
            float correct_class_probability = m_predictions[i][m_meta[i].klass];
            auto better = m_predictions[i] > correct_class_probability;
            int n_better = cuv::sum(better);
            if(n_better < k)
                correct++;
        }
        double loss = 1.0 - (correct / (double) cnt);
        LOG4CXX_WARN(g_log, "LOSS k="<<k<<", basis is prediction on "<< cnt<<" instances: " << loss);
        return loss;
    }

    void rgb_classification_dataset::set_image_basepath(std::string path){
        for(auto& m : m_meta){
            m.rgb_filename = path + "/" + m.rgb_filename;
        }
    }

    void rgb_classification_dataset::aggregate_statistics(const unsigned int n){
    }

}
