#ifndef __VOC_DETECTION_DATASET__
#     define __VOC_DETECTION_DATASET__
#include<boost/shared_ptr.hpp>
#include<cuv.hpp>
#include<string>
#include<vector>

namespace cuvnet
{
    /**
     * VOC object detection dataset.
     *
     * @note that this class if NOT derived from \c dataset: The whole dataset
     *       does not fit in RAM or would take too long to read...
     */
    class voc_detection_pipe;
    class voc_detection_dataset{
        public:
            struct object {
                unsigned int klass;  ///< the index of the class this object belongs to
                bool truncated;      ///< true if the truncated property was set in the XML file

                /// a 4-tuple of box coordinates
                /// @{
                unsigned int xmin;
                unsigned int xmax;
                unsigned int ymin;
                unsigned int ymax;
                /// @}
            };
            struct image_meta_info{
                std::string filename;   ///< image file name
                std::vector<object> objects; ///< descriptions of depicted objects

                /// a 4-tuple: coordinates of original image in squared image
                /// @{
                unsigned int xmin;
                unsigned int xmax;
                unsigned int ymin;
                unsigned int ymax;
                /// @}
            };

            /// a fully loaded pattern 
            struct pattern{
                voc_detection_dataset::image_meta_info meta_info;
                cuv::tensor<float,cuv::host_memory_space> img;
                cuv::tensor<float,cuv::host_memory_space> tch;
                cuv::tensor<float,cuv::host_memory_space> ign;
            };

        public:
            /**
             * constructor.
             *
             * Files should contain meta infos as produced by \c
             * util/voc_detection.py, which are summaries of the original VOC
             * XML files for easier C++ parsing.
             *
             * @param train_filename filename containing meta-infos for training set
             * @param test_filename filename containing meta-infos for test set
             * @param verbose if true, print number of images loaded
             */
            voc_detection_dataset(const std::string& train_filename, const std::string& test_filename, bool verbose=false);

            enum subset{
                SS_TRAIN, SS_VAL, SS_TEST
            };
            void switch_dataset(subset ss);

            unsigned int size_available()const;

            void get_batch(std::list<pattern>& dest, unsigned int n);

            unsigned int trainset_size()const{ return m_training_set.size(); }

        private:
            /**
             *
             * this reads meta infos as produced by \c util/voc_detection.py, which are summaries
             * of the original VOC XML files for easier C++ parsing.
             *
             */
            void read_meta_info(std::vector<image_meta_info>& dest, const std::string& filename, bool verbose=false);

            std::vector<image_meta_info> m_training_set;  ///< meta-infos of the training set
            std::vector<image_meta_info> m_val_set;       ///< meta-infos of the validation set
            std::vector<image_meta_info> m_test_set;      ///< meta-infos of the test set
            boost::shared_ptr<voc_detection_pipe> m_pipe; ///< a pipe providing access to loaded, pre-processed images
    };
}

#endif /* __VOC_DETECTION_DATASET__ */
