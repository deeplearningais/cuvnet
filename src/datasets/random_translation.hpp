
#ifndef __RANDOM_TRANSLATION_HPP__
#     define __RANDOM_TRANSLATION_HPP__
#include "dataset.hpp"
using namespace std;
namespace cuvnet
{
    /**
     * creates gauss filter \f[ exp(-distance^2 / sigma^2) \f]
     * @param gauss the gauss filter which is being created.
     * @param distance the distance of the Gauss filter
     * @param sigma sigma of the Gauss filter
     */
    void fill_gauss(cuv::tensor<float,cuv::host_memory_space> &gauss, int distance, int sigma);

    /**
     * convolves data with the gauss filter (smothing)
     * @param data data which is convolved with the kernel
     * @param kernel the kernel which is used to convolve with the data
     */
    void convolve_last_dim(cuv::tensor<float,cuv::host_memory_space>  &data, const cuv::tensor<float,cuv::host_memory_space>  & kernel);

    /**
     * subsamples the data
     * @param data the data which is being subsampled
     * @param each_elem subsamples each each_elem element
     *
     */
    void subsampling(cuv::tensor<float,cuv::host_memory_space>  &data, int each_elem);

    /**
     * Translate the data. Allows subpixel and full pixel translations.
     * @param data the data which is being translated
     * @param dim which dim is translated. If dim is 1 then the second dim is
     *        the translated version of the first. If dim is 0, then third dimension
     *        is translated version of the second dimension
     * @param trans_size how many elements to translate. If the number is
     *        positive, it translates the data to the right. If negative, to the left.
     *        If the translation index exceeds the border of the vector, then wrap
     *        around.
     */
    void translate_data(cuv::tensor<float,cuv::host_memory_space>  &data, int dim, const vector<int> &rand_translations);


    
/**
 * implements random translation of the vector. The train and test set consist of two inputs, where one input is translated version of
 * the other, and teacher which is translated version of the second input.
 *
 * @ingroup datasets
 */
    class random_translation: public dataset{
        private:
            int m_num_train_example;
            int m_num_test_example;
            int m_dim;
            float m_thres;
            int m_distance;
            float m_sigma;
        public:
        /**
         * Constructor
         *
         * @param dim dimension of the vector
         * @param num_train_examples the number of training examples 
         * @param num_test_examples the number of test examples 
         * @param thres the threshold for filling the train and test set with random numbers. If the random number generated is greater then threshold, then the one is assigned to the element of the data, otherwise zero is assigned. 
         * @param distance the distance of the gaussian filter
         * @param sigma sigma of the gaussian filter
         * @param subsample each subsample element is subsampled from the data.
         * @param translate_size how many elements to translate the data (with wrap-around).
         */
        random_translation(int dim, int num_train_examples, int num_test_examples, float thres, int distance, float sigma, int subsample, int translate_size);
        random_translation(){}
    };


}


#endif /* __RANDOM_TRANSLATION_HPP__ */
