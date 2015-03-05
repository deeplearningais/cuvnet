#include "cv_datasets.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace datasets
{
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

    std::vector<cv::RotatedRect> random_regions(const cv::Mat& img, int n_crops, float min_size_frac);
    std::vector<cv::RotatedRect> random_regions_from_depth(const cv::Mat& depth, int n_crops, double scale_fact, double min_size, double max_scale_fact=1.0);
    std::vector<cv::RotatedRect> covering_regions_from_depth(const cv::Mat& depth, double scale_fact, double min_size, double output_fraction);
    cv::Mat extract_region(const cv::Mat& m, const cv::RotatedRect& ir, bool flipped, int interpolation, int bordertype=cv::BORDER_REFLECT_101, int value=0);
    std::pair<cv::Rect, cv::RotatedRect>
        required_padding(const cv::Mat& m, const cv::RotatedRect& ir);
    void dstack_mat2tens(cuv::tensor<float, cuv::host_memory_space>& tens,
            const std::vector<cv::Mat>& src, bool reverse=false);
}
