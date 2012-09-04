
#ifndef __RANDOM_TRANSLATION_HPP__
#     define __RANDOM_TRANSLATION_HPP__
#include "dataset.hpp"
#include <vector>
namespace cuvnet
{
    /**
     * returns the wrap around index
     * @param size the size of the input
     * @param pos the current position in the input        
     * @return wrap around index
     */
    int get_wrap_index(int size, int pos);


    /**
     * performs growing transformation on data
     * @param data data which is initialized   
     * @param m_dim the size of the input
     * @param max_trans        
     * @param morse_factor the width of the morse input
     * @param max_grow the input can grow from -max_grow to +max_grow
     */
    void initialize_morse_code(cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels, int m_dim, int max_trans, int morse_factor, float max_grow);



    /**
     * performs growing transformation on data
     * @param data data on which the transformation is applied  
     * @param dim on which dim of data the transformation is applied 
     * @param translated true, if the translation transformation is applied on data
     * @param rand_growing consist the values of growing transformation which is applied on different examples
     */
    void growing_data(cuv::tensor<float,cuv::host_memory_space>  &data, int dim, bool translated, const std::vector<int> &rand_growing);
    

    /**
     * normalizes the data, such that values are between 0 and 1
     * @param data data which is normalized  
     */
    void normalize_data_set(cuv::tensor<float,cuv::host_memory_space> &data);



    /**
     * splits data into test and train set
     * @param data  data which is split
     * @param train_set train data which is initialized
     * @param test_set test data which is initialized
     */
    void split_data_set(cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels, cuv::tensor<float,cuv::host_memory_space>& train_set, cuv::tensor<float,cuv::host_memory_space>& test_set, cuv::tensor<float,cuv::host_memory_space>& train_labels, cuv::tensor<float,cuv::host_memory_space>& test_labels, int num_examples, int dim);


    /**
     * initializes the training and test set with bar of random width and position in the input
     * @param data  data which is initialized
     * @param m_dim the input size  
     * @param max_size the maximum size of the bar in the inputs
     * @param min_size the minimum size of the bar in the inputs
     * @param max_translation the maximum translation allowed
     *
     */
    void initialize_data_set_iter(int max_size, int min_size, cuv::tensor<float,cuv::host_memory_space>& data, int m_dim, int max_translation);


    /**
     * shuffles the examples in the dataset
     * @param data data which examples are shuffled
     */
    void shuffle(cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels);


    /**
     * initializes the training and test set
     * @param train_data train data which is initialized
     * @param test_data test data which is initialized
     * @param m_num_train_example the number of training examples which will be generated
     * @param m_num_test_example the number of test examples which will be generated
     * @param m_dim the input size  
     * @param m_thres the threshold for percentage of zeros in the input 
     * @param max_size the maximum size of the bar in the inputs
     * @param min_size the minimum size of the bar in the inputs
     * @param max_translation the maximum translation allowed
     * @param max_growing the maximum growing allowed
     * @param flag for value 0, the data set is initialized uniformly with m_thresh percentage of zeros. If 1, the data has random bars, and if 2, the data is morse code
     * @param morse_factor the width of the morse input
     *
     */
    void initialize_data_sets(cuv::tensor<float,cuv::host_memory_space>& train_data, cuv::tensor<float,cuv::host_memory_space>& test_data, 
            cuv::tensor<float,cuv::host_memory_space>& train_labels, cuv::tensor<float,cuv::host_memory_space>& test_labels,
            int m_num_train_example, int m_num_test_example, int m_dim, float m_thres, int max_size, int min_size, 
            int max_translation, float max_growing, int flag, int morse_factor);


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
    void translate_data(cuv::tensor<float,cuv::host_memory_space>  &data, int dim, const std::vector<int> &rand_translations);


    
/**
 * implements random translation of the vector. The train and test set consist of two inputs, where one input is translated version of
 * the other, and teacher which is translated version of the second input.
 *
 * @ingroup datasets
 */
    class random_translation: public dataset{
        private:
            int m_num_train_example;        ///< the number of training examples 
            int m_num_test_example;         ///< the number of test examples 
            int m_dim;                      ///< the size of the inputs 
            float m_thres;                  ///< the threshold for percentage of zeros in the input 
            int m_distance;                 ///< the distance used in smoothing (convolution) 
            float m_sigma;                  ///< the sigma used in smoothing (convolution)
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
         * @param flag indicates which input pattern types are created. if set to 0, inputs are randomly uniformly sampled. If 1, the bars of random width are created, and if set to 2, morse code is initialized
         * @param morse_factor the width of the morse input
         *
         */
        random_translation(int dim, int num_train_examples, int num_test_examples, float thres, int distance, float sigma, int subsample, int translate_size, float max_growing, int min_size, int max_size, int flag, int morse_factor);
        random_translation(){}
    };

/**
 * implements morse code dataset.  
 */
    class morse_code{
        private:
            // data where the code is written 
            cuv::tensor<float,cuv::host_memory_space> m_data;
            std::vector<std::vector<std::vector<float> > > m_coordinates;
            int m_factor;
            std::vector<std::string>  m_morse_code;
        public:
        /**
         * Constructor
         *
         * @param data the data where the morse code is stored
         * @param morse_factor the width of the morse input
         *
         */
            morse_code(cuv::tensor<float,cuv::host_memory_space> data, int factor);
            int get_wrap_index(int size, int pos);
            std::vector<std::string> get_morse_code();
            void init_morse_code_data_structure();
            int char_to_morse_index(char c);
            int get_width_char(int ch, int factor);
            int write_dot(int dim, int ex, float pos);
            int write_dash(int dim, int ex, float pos);
            int write_char(int ch, int dim, int ex, float pos);
            unsigned int get_size();
            cuv::tensor<float,cuv::host_memory_space> get_data();
            void write_from_coordinates();
            void write_bar(int dim, int ex, float coor_1, float coor_2);
            void translate_coordinates(int dim, int ex, int trans);       
            void scale_coordinates(int dim, int ex, float scale);
    };

}


#endif /* __RANDOM_TRANSLATION_HPP__ */
