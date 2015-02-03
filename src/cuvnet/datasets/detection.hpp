#ifndef __CUVNET_DETECTION_HPP__isfhaasdfhksjdhf
#     define __CUVNET_DETECTION_HPP__isfhaasdfhksjdhf
#include "cv_datasets.hpp"

namespace datasets
{
    /**
     * @addtogroup datasets
     * @{
     */

    /// a tag for object detection tasks
    struct rgb_detection_tag{};

    /// represents a bounding box of an object
    struct bbox{
        float x0, y0, w, h;
        int klass;
        float confidence;
        bool truncated;
    };
    
    struct rgb_detection_pattern : public rgb_pattern{
        std::vector<bbox> bboxes;
        std::vector<bbox> predicted_bboxes;
    };

    /// meta infos for detection
    template<>
        struct meta_data<rgb_detection_tag>{
            std::string rgb_filename;
            std::vector<bbox> bboxes;
            typedef rgb_image input_t;
            typedef pattern_with_original<rgb_detection_pattern, input_t> pattern_t;
            typedef pattern_t::patternset_t patternset_t;

            static void show(std::string name, const pattern_t& pat);
        };

    struct rgb_detection_dataset : public image_dataset<rgb_detection_tag> {
        typedef image_dataset<rgb_detection_tag> base_t;   ///< type of own base class
        rgb_detection_dataset(const std::string& filename, int pattern_size, int n_crops);
        
        /// the mean image, which is subtracted from all patterns before passing them to the model
        cuv::tensor<float, cuv::host_memory_space> m_imagenet_mean;

        /// number of crops generated for every image
        int m_n_crops;
        /// size of the crops generated
        int m_pattern_size;
        /// if true, cover image completely.
        bool m_exhaustive;

        /// number of different object classes
        unsigned int m_n_classes;
        /// the names of all the classes
        std::vector<std::string> m_class_names;
        /// @return number of different object classes
        inline unsigned int n_classes(){ return m_n_classes; }

        /// path to be prepended to paths in dataset file
        std::string m_image_basepath;

        /**
         * takes an image, crops (random) region(s) and returns the result as a cuv tensor.
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
    };

    /// @}
}
#endif /* __CUVNET_DETECTION_HPP__isfhaasdfhksjdhf */
