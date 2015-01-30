#ifndef __CV_DATASETS_HPP__
#     define __CV_DATASETS_HPP__
#include "pattern_set.hpp"

namespace cv
{
    RotatedRect;
}

namespace datasets{
    /// a wrapper for opencv's RotatedRect (so we don't need to include cv.h here)
    struct rotated_rect{
        int x, y, h, w;
        double a;
        rotated_rect& operator=(const cv::RotatedRect& r);
    };

    /// contains an RGB data
    struct rgb_image;

    /// contains an RGB data and depth data
    struct rgbd_image;

    /// contains an RGB data, depth data, and an image-like teacher
    struct rgbdt_image;

    /// contains an RGB data, and an image-like teacher
    struct rgbt_image;

    /// generic image pattern
    struct image_pattern{
        rotated_rect region_in_original;
        bool flipped;
    };

    /// contains RGB data and a class
    struct rgb_pattern : public image_pattern{
        boost::shared_ptr<rgb_classification_image> original;
        cuv::tensor<float, cuv::host_memory_space> rgb;
    };
    
    /// in addition to RGB data, contains depth information
    struct rgbd_pattern : rgb_pattern{
        boost::shared_ptr<rgbd_classification_image> original;
        cuv::tensor<float, cuv::host_memory_space> depth;
    };

    /// image classification (meant to be used with rgb_pattern or rgbd_pattern)
    template<class Base>
    struct image_classification_pattern : public Base{
        int klass;
    };

    /// object class segmentation (one label per pixel)
    template<class Base>
    struct image_objclassseg_pattern : public Base{
        cuv::tensor<float, cuv::host_memory_space> teacher;
        cuv::tensor<float, cuv::host_memory_space> ignoremask;
    };

    struct image_classification_dataset_impl;

    struct rgb_classification_tag{};
    struct rgbd_classification_tag{};
    struct rgb_objclassseg_tag{};
    struct rgbd_objclassseg_tag{};

    template<class C>
    struct meta_data { };
    template<>
        struct meta_data<rgb_classification_tag>{
            std::string rgb_filename;
            int klass;
            typedef rgb_image input_t;
            typedef image_classification_pattern<rgb_pattern> pattern_t;
            typedef pattern_set<pattern_t> patternset_t;
        };
    template<>
        struct meta_data<rgbd_classification_tag>{
            std::string rgb_filename;
            std::string depth_filename;
            int klass;
            typedef rgbd_image input_t;
            typedef image_classification_pattern<rgbd_pattern> pattern_t;
            typedef pattern_set<pattern_t> patternset_t;
        };
    template<>
        struct meta_data<rgb_objclassseg_tag>{
            std::string rgb_filename;
            std::string teacher_filename;
            typedef rgbt_image input_t;
            typedef image_objclassseg_pattern<rgb_pattern> pattern_t;
            typedef pattern_set<pattern_t> patternset_t;
        };
    template<>
        struct meta_data<rgbd_objclassseg_tag>{
            std::string rgb_filename;
            std::string depth_filename;
            std::string teacher_filename;
            typedef rgbdt_image input_t;
            typedef image_objclassseg_pattern<rgbd_pattern> pattern_t;
            typedef pattern_set<pattern_t> patternset_t;
        };

    template<class C>
    boost::shared_ptr<meta_data<C>::input_t> load_image(const meta_data<C>& data);

    /**
     * base class for datasets that need to be loaded from disk.
     *
     * Needs to be specialized by the user, especially the preprocess function,
     * and potentially the (global) load_image function.
     */
    template<class C>
    struct image_dataset{
        typedef meta_data<C> meta_t;
        typedef meta_t::input_t input_t;
        typedef meta_t::pattern_t pattern_t;
        typedef meta_t::patternset_t patternset_t;

        std::vector<meta_t> m_meta;

        virtual boost::shared_ptr<patternset_t> preprocess(const meta_t& meta, boost::shared_ptr<input_t> in) = 0;

        boost::shared_ptr<patternset_t> next(size_t idx){
            boost::shared_ptr<input_t> ptr = load_image(m_meta[idx]);
            return preprocess(m_meta[idx], ptr);
        }
    };
}

RotatedRect& operator=(const datasets::rotated_rect& r);

#endif /* __CV_DATASETS_HPP__ */
