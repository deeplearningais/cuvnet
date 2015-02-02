#ifndef __CV_DATASETS_HPP__
#     define __CV_DATASETS_HPP__
#include <cuv.hpp>
#include "pattern_set.hpp"

namespace cv
{
    struct RotatedRect;
}

namespace datasets{

    /*
     *  Guide to the classes in this file:
     *
     *  *_image: large stuff that is loaded from the disk more or less on demand
     *      the images are forward-declared only, since they contain OpenCV data structures which
     *      we don't want to include in this header.
     *
     *  *_pattern: possibly pre-processed images or bits of images that can be processed by a model
     *      naturally, they should be in the cuv tensor format.
     *
     *  *_tag: some tag that denotes a specific task you want to solve (which requires
     *       a certain combination of *_image, *_pattern and some meta-infos)
     *
     *  meta_data<tag>: a traits-class which specifies the meta-information required for a given tag
     *       meta-information is what is required to load a specific data point from disk,
     *       e.g. filenames. The structure also contains typedefs which specify which *_image and 
     *       *_pattern combination is used.
     */



    /// a wrapper for opencv's RotatedRect (so we don't need to include cv.h here)
    struct rotated_rect{
        float x, y, h, w;
        double a;
        rotated_rect& operator=(const cv::RotatedRect& r);
        operator cv::RotatedRect()const;
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

    /// contains RGB data
    struct rgb_pattern : public image_pattern{
        boost::shared_ptr<rgb_image> original;
        cuv::tensor<float, cuv::host_memory_space> rgb;
    };
    
    /// in addition to RGB data, contains depth information
    struct rgbd_pattern : rgb_pattern{
        boost::shared_ptr<rgbd_image> original;
        cuv::tensor<float, cuv::host_memory_space> depth;
    };

    /// image classification (meant to be used with rgb_pattern or rgbd_pattern)
    template<class Base>
    struct image_classification_pattern : public Base{
        int ground_truth_class;
        cuv::tensor<float, cuv::host_memory_space> predicted_class;
    };

    /// object class segmentation (one label per pixel)
    template<class Base>
    struct image_objclassseg_pattern : public Base{
        cuv::tensor<float, cuv::host_memory_space> teacher;
        cuv::tensor<float, cuv::host_memory_space> ignoremask;

        cuv::tensor<float, cuv::host_memory_space> predicted;
    };

    /// link pattern to its original image
    template<class Pattern, class Input>
    struct pattern_with_original : public Pattern{
        typedef pattern_set<pattern_with_original<Pattern,Input> > patternset_t;
        boost::shared_ptr<Input> original;
        boost::shared_ptr<patternset_t> set;
    };

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
            typedef pattern_with_original<image_classification_pattern<rgb_pattern>, input_t> pattern_t;
            typedef pattern_t::patternset_t patternset_t;
        };
    template<>
        struct meta_data<rgbd_classification_tag>{
            std::string rgb_filename;
            std::string depth_filename;
            int klass;
            typedef rgbd_image input_t;
            typedef pattern_with_original<image_classification_pattern<rgbd_pattern>, input_t> pattern_t;
            typedef pattern_t::patternset_t patternset_t;
        };
    template<>
        struct meta_data<rgb_objclassseg_tag>{
            std::string rgb_filename;
            std::string teacher_filename;
            typedef rgbt_image input_t;
            typedef pattern_with_original<image_objclassseg_pattern<rgb_pattern>, input_t> pattern_t;
            typedef pattern_t::patternset_t patternset_t;
        };
    template<>
        struct meta_data<rgbd_objclassseg_tag>{
            std::string rgb_filename;
            std::string depth_filename;
            std::string teacher_filename;
            typedef rgbdt_image input_t;
            typedef pattern_with_original<image_objclassseg_pattern<rgbd_pattern>, input_t> pattern_t;
            typedef pattern_t::patternset_t patternset_t;
        };

    template<class C>
    boost::shared_ptr<typename meta_data<C>::input_t> load_image(const meta_data<C>& data);

    /**
     * base class for datasets that need to be loaded from disk.
     *
     * Needs to be specialized by the user, especially the preprocess function,
     * and potentially the (global) load_image function.
     */
    template<class C>
    struct image_dataset{
        typedef meta_data<C> meta_t;
        typedef typename meta_t::input_t input_t;
        typedef typename meta_t::pattern_t pattern_t;
        typedef typename meta_t::patternset_t patternset_t;

        std::vector<meta_t> m_meta;

        inline size_t size()const{ return m_meta.size(); }

        virtual boost::shared_ptr<patternset_t> preprocess(size_t idx, boost::shared_ptr<input_t> in) = 0;
        virtual void notify_done(boost::shared_ptr<pattern_t> pat){
            pat->set->notify_processed(pat);
        }

        boost::shared_ptr<patternset_t> next(size_t idx){
            boost::shared_ptr<input_t> ptr = load_image(m_meta[idx]);
            return preprocess(idx, ptr);
        }
    };


    /**
     * sample implementation of a dataset where every RGB image is associated with a single class label.
     *
     * Every image can be processed in multiple "crops".
     * The predictions over crops are averaged.
     */
    struct rgb_classification_dataset : public image_dataset<rgb_classification_tag> {
        typedef image_dataset<rgb_classification_tag> base_t;
        int m_n_crops;
        int m_pattern_size;
        int m_n_classes;
        std::vector<std::string> m_class_names;
        std::vector<int>  m_predictions;
        cuv::tensor<float, cuv::host_memory_space> m_imagenet_mean;
        rgb_classification_dataset(const std::string& filename, int pattern_size, int n_crops);
        boost::shared_ptr<patternset_t> preprocess(size_t idx, boost::shared_ptr<input_t> in) override;
        void notify_done(boost::shared_ptr<pattern_t> pat) override;
    }; 
}


#endif /* __CV_DATASETS_HPP__ */
