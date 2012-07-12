
#include <vector>
#include <cmath>
#include <cuv.hpp>
#include "random_translation.hpp"
#include <algorithm>
using namespace std;
namespace cuvnet
{
        random_translation::random_translation(int dim, int num_train_examples, int num_test_examples, float thres, int distance, float sigma, int subsample, int max_translation, int min_size, int max_size):
            m_num_train_example(num_train_examples),
            m_num_test_example(num_test_examples),
            m_dim(dim),
            m_thres(thres),
            m_distance(distance),
            m_sigma(sigma)
        {
            srand ( time(NULL) );
            train_data.resize(cuv::extents[3][m_num_train_example][m_dim]);
            test_data.resize(cuv::extents[3][m_num_test_example][m_dim]);
            
            // fills the train and test sets with random uniform numbers
           
            //cuv::fill_rnd_uniform(train_data);
            //cuv::fill_rnd_uniform(test_data);
            //cuv::apply_scalar_functor(train_data,cuv::SF_LT,m_thres);
            //cuv::apply_scalar_functor(test_data,cuv::SF_LT,m_thres);
          
            
            // initializes the data in the way that ones are next to each other
            
            int random_elem;
            int diff = max_size - min_size + 1;
            int wrap_index;
            int max_index;
            for(unsigned int ex = 0; ex < train_data.shape(1); ex++){
               random_elem = rand() % m_dim;
               //makes the size of the ones between the min_size and max_size
               int size = rand() % diff + min_size;  
               //int size = max_size  ;  
               max_index = random_elem + size;
               if(max_index >= m_dim)
                   wrap_index = max_index - m_dim;
               else
                   wrap_index = -1;

               for(int elem = 0; elem < m_dim; elem++){
                   if((elem >= random_elem && elem <= max_index) || (elem <= wrap_index) ){
                       train_data(0,ex,elem) = 1;
                   }else
                       train_data(0,ex, elem) = 0;
               }
            }


            // creates the vector for random translation. It is used to randomly translate each example vector
            vector<int> random_translations_train(train_data.shape(1));
            for(unsigned int i = 0; i < train_data.shape(1); i++){
                // For each example, the random translation is a number from  [- max_translation, + max_translation]
                random_translations_train[i] = rand() % (2 * max_translation  + 1) - max_translation;
            }

            vector<int> random_translations_test(test_data.shape(1));
            for(unsigned int i = 0; i < test_data.shape(1); i++){
                // For each example, the random translation is a number from  [- max_translation, + max_translation]
                random_translations_test[i] = rand() % (2 * max_translation  + 1) - max_translation;
            }


            // translate train data 
            translate_data(train_data, 1, random_translations_train);
            translate_data(train_data, 2, random_translations_train);

            // translate test data 
            translate_data(test_data, 1, random_translations_test);
            translate_data(test_data, 2, random_translations_test);

            // creates gaussian filter
            cuv::tensor<float,cuv::host_memory_space> gauss;
            fill_gauss(gauss, m_distance, m_sigma);

            // convolves last dim of both train and test data with the gauss filter
            convolve_last_dim(train_data, gauss);
            convolve_last_dim(test_data, gauss);

            std::cout << " train data dim before subsampling : " << train_data.shape(2) << std::endl;
            // subsamples each "subsample" element
            subsampling(train_data, subsample);
            subsampling(test_data,subsample);
            std::cout << " train data dim after subsampling : " << train_data.shape(2) << std::endl;


            normalize_data_set(train_data);
            normalize_data_set(test_data); 
            std::cout << "mean after subs: "<< cuv::mean(train_data)<<std::endl;
            std::cout << "var : "<< cuv::var(train_data)<<std::endl;

        }


        void normalize_data_set(cuv::tensor<float,cuv::host_memory_space>& data){
            using namespace cuv;
            typedef cuv::tensor<float,cuv::host_memory_space> tens_t;
            tens_t tmp;
            float max_1, max_2, max_3, min_1, min_2, min_3, min_, max_ = 0;
            for(unsigned int ex = 0; ex < data.shape(1); ex++){
               
               
                min_1 = minimum(data[indices[0][ex][index_range()]]);
                min_2 = minimum(data[indices[1][ex][index_range()]]);
                min_3 = minimum(data[indices[2][ex][index_range()]]);
                min_ = min(min_1, min_2);
                min_ = min(min_, min_3);

                tmp = data[indices[0][ex][index_range()]];
                tmp -= min_;
                tmp = data[indices[1][ex][index_range()]];
                tmp -= min_;
                tmp = data[indices[2][ex][index_range()]];
                tmp -= min_;
                
                max_1 = maximum(data[indices[0][ex][index_range()]]);
                max_2 = maximum(data[indices[1][ex][index_range()]]);
                max_3 = maximum(data[indices[2][ex][index_range()]]);
                max_ = max(max_1, max_2);
                max_ = max(max_, max_3);
                
                if(max_ <  0.00001f)
                    max_ = 0.00001f;

                tmp = data[indices[0][ex][index_range()]];
                tmp /= max_;
                tmp = data[indices[1][ex][index_range()]];
                tmp /= max_;
                tmp = data[indices[2][ex][index_range()]];
                tmp /= max_;
            }   
        }

        void fill_gauss(cuv::tensor<float,cuv::host_memory_space> &gauss, int distance, int sigma){
            gauss.resize(cuv::extents[2 * distance + 1]);
            for(int d = - distance ; d <= distance; d++){
                gauss(d + distance) = std::exp(- (float)d*d / (sigma * sigma));
            }
            // normalize gauss filters, that all values sum to 1
            gauss /= cuv::sum(gauss);
        }


        void convolve_last_dim(cuv::tensor<float,cuv::host_memory_space>  &data, const cuv::tensor<float,cuv::host_memory_space>  & kernel){
            int dim = data.shape(2);
            cuv::tensor<float,cuv::host_memory_space>  temp_data(data.copy());
            for(int t = 0; t < (int)data.shape(0); t++){
                for (int ex = 0; ex < (int)data.shape(1); ex++){
                    for(int d = 0; d < dim; d++){
                        float sum = 0;  
                        int distance = (kernel.size() - 1) / 2;
                        for(int dist = - distance; dist <= distance; dist++){
                            int wrap_index = d - dist;
                            // do wrap around if the index is negative
                            if(wrap_index < 0)
                                // go to the right side 
                                wrap_index = dim + wrap_index;
                            else if(wrap_index >= dim)
                                // go to the left side
                                wrap_index = dim - (wrap_index);        
                            sum +=  data(t, ex, wrap_index) * kernel(dist + distance);
                            
                        }
                        // update the element
                        temp_data(t, ex, d) = sum;
                    }
                } 
            }
            data = temp_data.copy();
        }
        

        void subsampling(cuv::tensor<float,cuv::host_memory_space>  &data, int each_elem){
            assert(data.shape(2) % each_elem == 0);
            cuv::tensor<float,cuv::host_memory_space>  tmp_data(cuv::extents[data.shape(0)][data.shape(1)][data.shape(2) / each_elem]);
            for(unsigned int i = 0; i < tmp_data.shape(0); i++){
                for(unsigned int j = 0; j < tmp_data.shape(1); j++){
                    for(unsigned int k = 0; k < tmp_data.shape(2); k++){
                        tmp_data(i, j, k) = data(i, j, k * each_elem);
                    }
                }
            }
            data.resize(cuv::extents[data.shape(0)][data.shape(1)][data.shape(2) / each_elem]);
            data = tmp_data.copy();
        }
        

        
        // the second dim is the translated version of the first,
        // and third dimension is translated version of the second
        void translate_data(cuv::tensor<float,cuv::host_memory_space>  &data, int dim, const vector<int> &rand_translations){
            assert(dim == 1 || dim == 2);
            for(unsigned int i = 0; i < data.shape(1); i++){
                for(unsigned int j = 0; j < data.shape(2); j++){
                    int index = j - rand_translations[i];
                    // wrap around if the index goes out of border
                    if(index < 0)
                        index = data.shape(2) + index;
                    else if(index >= (int)data.shape(2))
                        index = index - data.shape(2);
                    data(dim, i, j) = data(dim - 1, i, index);
                }
            }
        }

}


