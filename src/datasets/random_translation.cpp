
#include <vector>
#include <cmath>
#include <cuv.hpp>
#include "random_translation.hpp"

using namespace std;
namespace cuvnet
{
        random_translation::random_translation(int dim, int num_train_examples, int num_test_examples, float thres, int distance, float sigma, int subsample, int translate_size):
            m_num_train_example(num_train_examples),
            m_num_test_example(num_test_examples),
            m_dim(dim),
            m_thres(thres),
            m_distance(distance),
            m_sigma(sigma)
        {
            train_data.resize(cuv::extents[m_num_train_example][m_dim][3]);
            test_data.resize(cuv::extents[m_num_test_example][m_dim][3]);
            
            // fills the train and test sets with random uniform numbers
            cuv::fill_rnd_uniform(train_data);
            cuv::fill_rnd_uniform(test_data);
            cuv::apply_scalar_functor(train_data,cuv::SF_LT,m_thres);
            cuv::apply_scalar_functor(test_data,cuv::SF_LT,m_thres);
            
            // translate train data 
            translate_data(train_data, 1, translate_size);
            translate_data(train_data, 2, translate_size);

            // translate test data 
            translate_data(test_data, 1, translate_size);
            translate_data(test_data, 2, translate_size);


            // creates gaussian filter
            cuv::tensor<float,cuv::host_memory_space> gauss;
            fill_gauss(gauss, m_distance, m_sigma);

            // convolves last dim of both train and test data with the gauss filter
            convolve_last_dim(train_data, gauss);
            convolve_last_dim(test_data, gauss);
            
            // subsamples each "subsample" element
            subsampling(train_data, subsample);
            subsampling(test_data,subsample);

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
            int dim = data.shape(1);
            cuv::tensor<float,cuv::host_memory_space>  temp_data(data.copy());
            for (int ex = 0; ex < (int)data.shape(0); ex++){
                for(int d = 0; d < dim; d++){
                    for(int t = 0; t < (int)data.shape(2); t++){
                         
                        int sum = 0;  
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
                            sum +=  data(ex, wrap_index, t) * kernel(dist + distance); 
                        }
                        // update the element
                        temp_data(ex, d ,t) = sum;
                    }
                } 
            }
            data = temp_data.copy();
        }
        

        void subsampling(cuv::tensor<float,cuv::host_memory_space>  &data, int each_elem){
            assert(data.shape(1) % each_elem == 0);
            cuv::tensor<float,cuv::host_memory_space>  tmp_data(cuv::extents[data.shape(0)][data.shape(1) / each_elem][data.shape(2)]);
            for(int i = 0; i < tmp_data.shape(0); i++){
                for(int j = 0; j < tmp_data.shape(1); j++){
                    for(int k = 0; k < tmp_data.shape(2); k++){
                        tmp_data(i,j,k) = data(i,j * each_elem,k);
                    }
                }
            }
            data.resize(cuv::extents[data.shape(0)][data.shape(1) / each_elem][data.shape(2)]);
            data = tmp_data.copy();
        }
        

        
        void translate_data(cuv::tensor<float,cuv::host_memory_space>  &data, int dim, int trans_size){
            assert(dim == 1 || dim == 2);
            for(int i = 0; i < data.shape(0); i++){
                for(int j = 0; j < data.shape(1); j++){
                    // the second dim is the translated version of the first
                    int index = j - trans_size;
                    // wrap around if the index goes out of border
                    if(index < 0)
                        index = data.shape(1) + index;
                    else if(index >= data.shape(1))
                        index = index - data.shape(1);
                    data(i, j, dim) = data(i, index, dim - 1);
                }
            }
        }

}


