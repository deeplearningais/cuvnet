#ifndef __CV_DATASETS_HPP__
#     define __CV_DATASETS_HPP__
#include <cuv.hpp>
#include "pattern_set.hpp"

namespace cv
{
    //struct Rect;
    template<class T>
        struct Rect_;
    typedef Rect_<int> Rect;
    struct RotatedRect;
    class Mat;
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



    /** @addtogroup datasets
     * @{
     */

    /**
     * a wrapper for opencv's RotatedRect (so we don't need to include cv.h here)
     */
    struct rotated_rect{
        float x, y, h, w;
        double a;
        rotated_rect& operator=(const cv::RotatedRect& r);
        operator cv::RotatedRect()const;
        rotated_rect(){};
        rotated_rect(float x, float y, float h, float w, double a)
            :x(x), y(y), h(h), w(w), a(a){};

        /// calculates the l2 distance of two rectangles. Warning: Ignores rotation!
        static inline float l2dist(const rotated_rect& a, const rotated_rect& b) {
            return std::sqrt(
                    std::pow(a.x - b.x, 2) +
                    std::pow(a.y - b.y, 2) +
                    std::pow(a.h - b.h, 2) +
                    std::pow(a.w - b.w, 2)
                    );
        };
        
        friend rotated_rect operator+(const rotated_rect& l, const rotated_rect& r) {
           return rotated_rect(
                   l.x + r.x,
                   l.y + r.y,
                   l.h + r.h,
                   l.w + r.w,
                   l.a + r.a
                   ); 
        };
        friend rotated_rect operator-(const rotated_rect& l, const rotated_rect& r) {
           return rotated_rect(
                   l.x - r.x,
                   l.y - r.y,
                   l.h - r.h,
                   l.w - r.w,
                   l.a - r.a
                   ); 
        };

        /// scales parameters independly by a factor. Useful for a scaled difference of rectangles. 
        // Ignores rotation
        inline rotated_rect& scale_like_vec(const float f) {
            x *= f;
            y *= f;
            h *= f;
            w *= f;
            return *this;
        };

        // necessary if one wants to export vectors of vectors of this struct to python
        bool operator==(const rotated_rect& b)const{
            const rotated_rect& a = *this;
            return (a.x == b.x) && (a.y == b.y) && (a.h == b.h) && (a.w == b.w) && (a.a == b.a);
        }
        bool operator!=(const rotated_rect& b)const{
            const rotated_rect& a = *this;
            return (a.x != b.x) || (a.y != b.y) || (a.h != b.h) || (a.w != b.w) || (a.a != b.a);
        }

        private:
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version){
                    ar & x;
                    ar & y;
                    ar & h;
                    ar & w;
                    ar & a;
                }
    };


    /**
     * @name Image Types
     *  Large stuff that is loaded from the disk more or less on demand
     *  the images are forward-declared only, since they contain OpenCV data structures which
     *  we don't want to include in this header.
     */
    ///@{
    
    /**
     * Contains RGB data for an image.
     */
    struct rgb_image;

    /**
     * Contains RGB data and depth data
     */
    struct rgbd_image;

    /// Contains RGB data, depth data, and an image-like teacher
    struct rgbdt_image;

    /// Contains RGB data, and an image-like teacher
    struct rgbt_image;
    ///@}

    /**
     * @name Pattern Types
     *  Possibly pre-processed images or bits of images that can be processed
     *  by a model naturally, they should be in the cuv tensor format.
     */
    ///@{
    /// generic image pattern
    struct image_pattern{
        rotated_rect region_in_original;
        bool flipped;
    };

    /// contains RGB data
    struct rgb_pattern : public image_pattern{
        cuv::tensor<float, cuv::host_memory_space> rgb;
    };
    
    /// in addition to RGB data, contains depth information
    struct rgbd_pattern : rgb_pattern{
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
    ///@}

    /**
     * @name Task tags
     *  Some tag that denotes a specific task you want to solve (which requires
     *  a certain combination of *_image, *_pattern and some meta-infos).
     *  The tag is empty, information about it is carried by the meta_data
     *  class.
     */
    ///@{
    struct rgb_classification_tag{};
    struct rgbd_classification_tag{};
    struct rgb_objclassseg_tag{};
    struct rgbd_objclassseg_tag{};
    ///@}

    /**
     * @name Meta-Data
     *  a traits-class which specifies the meta-information required for a given tag
     *  meta-information is what is required to load a specific data point from disk,
     *  eg filenames. 
     *  
     *  The structure also contains typedefs which specify the *_image and
     *  *_pattern combination you want to use.
     */
    ///@{
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
    ///@}

    /**
     * Load an image from disc.
     * @param data contains filenames etc, ie information that fits into RAM.
     */
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

        /// contains information about the dataset elements that fits into RAM.
        std::vector<meta_t> m_meta;

        /// contains a mapping from virtual indices to indices in m_meta
        std::vector<size_t> m_shuffled_idx;

        /// shuffle access to the dataset
        void shuffle(bool really=true){
            if(m_shuffled_idx.size() != m_meta.size()){
                m_shuffled_idx.resize(m_meta.size());
                for (size_t i = 0; i < m_meta.size(); ++i)
                    m_shuffled_idx[i] = i;
            }
            if(really)
                std::random_shuffle(m_shuffled_idx.begin(), m_shuffled_idx.end());
        }

        /// @return the number of elements in the dataset
        inline size_t size()const{ return m_meta.size(); }

        /**
         * This function converts a file loaded from disk into one or more patterns to be processed by a model.
         * Note that it is const to ensure that it is thread-safe, it should not have any side-effects.
         */
        virtual boost::shared_ptr<patternset_t> preprocess(size_t idx, boost::shared_ptr<input_t> in) const = 0;
        virtual void notify_done(boost::shared_ptr<pattern_t> pat){
            pat->set->notify_processed(pat);
        }

        /// Used by image_queue to fetch a specific element from the dataset.
        boost::shared_ptr<patternset_t> next(size_t idx)const{
            boost::shared_ptr<input_t> ptr = load_image(m_meta[m_shuffled_idx[idx]]);
            return preprocess(m_shuffled_idx[idx], ptr);
        }
    };

    /**
     * sample implementation of a dataset where every RGB image is associated with a single class label.
     *
     * Every image can be processed in multiple "crops".
     * The predictions over crops are averaged.
     */
    struct rgb_classification_dataset : public image_dataset<rgb_classification_tag> {
        typedef image_dataset<rgb_classification_tag> base_t;   ///< type of own base class
        /// number of crops generated for every image
        int m_n_crops;
        /// size of the crops generated
        int m_pattern_size;
        /// @return number of classes this dataset contains
        inline unsigned int n_classes(){ return m_class_names.size(); }
        /// all class names in the dataset
        std::vector<std::string> m_class_names;
        /// the predicted classes for every element in the dataset
        std::vector<cuv::tensor<float, cuv::host_memory_space> >  m_predictions;
        /// the mean image, which is subtracted from all patterns before passing them to the model
        cuv::tensor<float, cuv::host_memory_space> m_imagenet_mean;
        /// path to be prepended to paths in dataset file
        std::string m_image_basepath;
        /**
         * ctor. 
         * @param filename the filename containing the dataset.
         *                 The file should contain lines with image filename, dummy int, class index, separated by spaces.
         * @param pattern_size size of the crops out of the original image
         * @param n_crops how many crops to generate for every image loaded
         */
        rgb_classification_dataset(const std::string& filename, int pattern_size, int n_crops);
        /**
         * takes an image, crops random region(s) and returns the result as a cuv tensor.
         */
        boost::shared_ptr<patternset_t> preprocess(size_t idx, boost::shared_ptr<input_t> in) const override;

        /**
         * Records the model prediction.
         */
        void notify_done(boost::shared_ptr<pattern_t> pat) override;

        /**
         * set a binary file containing a 3xNxN float32 C-ordered imagenet mean.
         *
         * where N is the pattern_size supplied to the constructor.
         */
        void set_imagenet_mean(std::string filename);

        /**
         * Set path prepended to path in dataset file.
         * @note this is destructive and cannot be called twice.
         */
        void set_image_basepath(std::string path);

        /**
         * clears all predictions.
         */
        void clear_predictions();

        /**
         * determine the one loss over the prediction
         * @param k instance is classified 'correctly' if ground_truth_class is
         *          among the k predictions with highest confidence
         */
        float get_zero_one(int k=1);
    }; 

    /** 
     * @}
     */
}


#endif /* __CV_DATASETS_HPP__ */
