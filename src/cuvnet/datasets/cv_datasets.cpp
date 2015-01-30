#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cv_datasets.hpp"

namespace datasets
{
    
    struct rgb_image{
        cv::Mat rgb;
    };

    struct rgbd_image : public rgb_image{
        cv::Mat depth, height;
    };

    struct rgbdt_image : public rgbd_image{
        cv::vector<cv::Mat> cls;
        cv::vector<cv::Mat> prediction;
    };

    template<>
    boost::shared_ptr<meta_data<rgb_objclassseg_tag>::input_t> load_image(const meta_data<rgb_objclassseg_tag>& meta){
        auto img = boost::make_shared<rgbt_image>();
        img->rgb = cv::imread(meta.rgb, CV_LOAD_IMAGE_COLOR);
        img->ID = rgb;
        cv::cvtColor(img.rgb, img.rgb, CV_BGR2RGB);
        {
            int marg = 9; // 9
            img.rgb = img.rgb(cv::Range(marg, img.rgb.rows-marg), cv::Range(marg, img.rgb.cols-marg)).clone();
            copyMakeBorder(img.rgb, img.rgb, marg, marg, marg, marg, cv::BORDER_REFLECT_101, 0);
        }

        img.cls.resize(4);
        img.ign = cv::Mat::ones(img.rgb.rows, img.rgb.cols, CV_32FC1);
        cv::Mat m = cv::imread(cls.c_str(), -1);
        for(auto& tmp : img.cls)
            tmp = cv::Mat::zeros(img.rgb.rows, img.rgb.cols, CV_32FC1);
        unsigned short* p = (unsigned short*)m.data;
        for(unsigned int y=0; y<m.rows; y++){
            for(unsigned int x=0; x<m.cols; x++){
                int v = *p;
                assert(v == 255 || v <= 4);
                if(v == 255)
                    img.ign.at<float>(y,x) = 0;
                else
                    img.cls[v].at<float>(y,x) = 1;
                p++;
            }
        }
        return img;
    }
    rgbd_objclass_image   load_objclass_image(const std::string& rgb, const std::string& cls, const std::string& depth){
        rgbd_objclass_image  img;
        {
            rgb_objclass_image tmp = load_objclass_image(rgb,cls);
            img.ID  = tmp.ID;
            img.rgb = tmp.rgb;
            img.cls = tmp.cls;
            img.ign = tmp.ign;
        }

        cnpy::NpyArray npDepth = cnpy::npy_load(depth);
        img.depth = cv::Mat (npDepth.shape[1], npDepth.shape[0], CV_32F);
        assert(img.depth.isContinuous());

        memcpy(img.depth.data, npDepth.data, sizeof(float)*img.depth.rows*img.depth.cols);
        cv::transpose(img.depth, img.depth);
        assert(img.depth.isContinuous());
        npDepth.destruct(); // remove numpy array from memory
        {
            int marg = 6; // was 5
            img.depth = img.depth(cv::Range(marg, img.rgb.rows-marg), cv::Range(marg, img.rgb.cols-marg)).clone();
            cv::copyMakeBorder(img.depth, img.depth, marg, marg, marg, marg, cv::BORDER_REFLECT_101);
        }
        img.height = cv::Mat(480, 640, CV_32FC1);
        std::ifstream ifs(depth.substr(0, depth.size()-3) + "height");
        ifs.read((char*)img.height.data, 640*480*sizeof(float));
        return img;
    };
}
