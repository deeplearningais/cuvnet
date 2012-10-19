#ifndef __VOC_DETECTION_DATASET__
#     define __VOC_DETECTION_DATASET__
#include<boost/shared_ptr.hpp>
#include<cuv.hpp>
#include<string>
#include<vector>
#include"bounding_box_tools.hpp"

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

            /// a fully loaded pattern.
            struct pattern{
                bbtools::image_meta_info meta_info;
                cuv::tensor<float,cuv::host_memory_space> img; ///< image
                cuv::tensor<float,cuv::host_memory_space> tch; ///< teacher
                cuv::tensor<float,cuv::host_memory_space> ign; ///< ignore mask

                cuv::tensor<float,cuv::host_memory_space> result; ///< result
            };

            /** 
             * assuming the input tensors are of shape NxN, the teacher/ignore
             * tensors are of shape MxM, where M = (N-crop)/scale.
             * This is useful if the teacher/ignore variables are to be used in
             * a network that performs valid convolutions (crop) and pooling
             * (scale) operations.
             */
            struct output_properties{
                unsigned int scale_h, scale_w, crop_h, crop_w;
            };
            
            output_properties m_output_properties;

            void set_output_properties(
                    unsigned int scale_h, unsigned int scale_w,
                    unsigned int crop_h, unsigned int crop_w);
            
        private:
            int m_n_threads;
            std::list<pattern> m_return_queue;


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
             * @param n_threads number of threads to start for queue. If zero, use number of CPUs
             * @param verbose if true, print number of images loaded
             */
            voc_detection_dataset(const std::string& train_filename, const std::string& test_filename, int n_threads=0, bool verbose=false);

            enum subset{
                SS_TRAIN, SS_VAL, SS_TEST
            };
            /**
             * change the subset of the dataset .
             *
             * @param ss the new subset
             * @param n_threads if it is zero, use value given in constructor
             *
             */
            void switch_dataset(subset ss, int n_threads=0);

            /// number of elements currently in the queue waiting to be processed
            unsigned int size_available()const;

            /// get a batch of n elements
            void get_batch(std::list<pattern>& dest, unsigned int n);

            /// return the number of images in the training set
            unsigned int trainset_size()const{ return m_training_set.size(); }

            /// save results for processing (eg by combining cropped subimages)
            void save_results(std::list<pattern>& results);

        private:
            /**
             *
             * this reads meta infos as produced by \c util/voc_detection.py, which are summaries
             * of the original VOC XML files for easier C++ parsing.
             *
             */
            void read_meta_info(std::vector<bbtools::image_meta_info>& dest, const std::string& filename, bool verbose=false);

            std::vector<bbtools::image_meta_info> m_training_set;  ///< meta-infos of the training set
            std::vector<bbtools::image_meta_info> m_val_set;       ///< meta-infos of the validation set
            std::vector<bbtools::image_meta_info> m_test_set;      ///< meta-infos of the test set
            boost::shared_ptr<voc_detection_pipe> m_pipe; ///< a pipe providing access to loaded, pre-processed images
    };
}

#endif /* __VOC_DETECTION_DATASET__ */
