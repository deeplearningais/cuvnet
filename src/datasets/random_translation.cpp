
#include <vector>
#include <cmath>
#include <cuv.hpp>
#include "random_translation.hpp"
#include <algorithm>
using namespace std;
namespace cuvnet
{

class Morse_code{
    private:
        // data where the code is written 
        cuv::tensor<float,cuv::host_memory_space> data;
    public:
        // constructor
        Morse_code(cuv::tensor<float,cuv::host_memory_space> data_):
        data(data_.copy())
        {
        }

        // returns wrap around index
        int get_wrap_index(int size, int pos){
            if(pos >= size){
                return pos - size;
            }else{
                return pos;
            }
        }

        // writes value 1 at position pos, and 0 after it
        int write_dot(int dim, int ex, int pos){
            int size = data.shape(2);
            data(dim,ex, pos) = 1.f;
            data(dim,ex, get_wrap_index(size, pos + 1)) = 0.f;
            return get_wrap_index(size, pos + 2);
        }

        // writes at position pos 3 times 1 and once 0
        int write_dash(int dim, int ex, int pos){
            int size = data.shape(2);
            data(dim,ex, pos) = 1.f;
            data(dim,ex, get_wrap_index(size, pos + 1)) = 1.f;
            data(dim,ex, get_wrap_index(size, pos + 2)) = 1.f;
            data(dim,ex, get_wrap_index(size, pos + 3)) = 0.f;
            return get_wrap_index(size,pos + 4);
        }

        // writes letter A and returns the index where the cursor is 
        int write_a(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
            
        }

        // writes letter B and returns the index where the cursor is 
        int write_b(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }
        // writes letter C and returns the index where the cursor is 
        int write_c(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }
        // writes letter D and returns the index where the cursor is 
        int write_d(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }
        
        // writes letter E and returns the index where the cursor is 
        int write_e(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }

        // writes letter F and returns the index where the cursor is 
        int write_f(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }


        // writes letter G and returns the index where the cursor is 
        int write_g(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }


        // writes letter H and returns the index where the cursor is 
        int write_h(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }


        // writes letter I and returns the index where the cursor is 
        int write_i(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }

        // writes letter J and returns the index where the cursor is 
        int write_j(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter K and returns the index where the cursor is 
        int write_k(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter l and returns the index where the cursor is 
        int write_l(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }


        // writes letter M and returns the index where the cursor is 
        int write_m(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }
        // writes letter N and returns the index where the cursor is 
        int write_n(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }
        // writes letter o and returns the index where the cursor is 
        int write_o(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }
        // writes letter p and returns the index where the cursor is 
        int write_p(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }
        // writes letter Q and returns the index where the cursor is 
        int write_q(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }
        // writes letter R and returns the index where the cursor is 
        int write_r(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }
        // writes letter S and returns the index where the cursor is 
        int write_s(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }
        // writes letter T and returns the index where the cursor is 
        int write_t(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter U and returns the index where the cursor is 
        int write_u(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }

        // writes letter V and returns the index where the cursor is 
        int write_v(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }
        // writes letter W and returns the index where the cursor is 
        int write_w(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }

        // writes letter X and returns the index where the cursor is 
        int write_x(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter Y and returns the index where the cursor is 
        int write_y(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter Z and returns the index where the cursor is 
        int write_z(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }

        // writes letter 0 and returns the index where the cursor is 
        int write_0(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }




        // writes letter  and returns the index where the cursor is 
        int write_1(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter 2 and returns the index where the cursor is 
        int write_2(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter 3 and returns the index where the cursor is 
        int write_3(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }


        // writes letter 4 and returns the index where the cursor is 
        int write_4(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            return pose_index;
        }



        // writes letter 5 and returns the index where the cursor is 
        int write_5(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }

        // writes letter 6 and returns the index where the cursor is 
        int write_6(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }

        // writes letter 7 and returns the index where the cursor is 
        int write_7(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }

        // writes letter 8 and returns the index where the cursor is 
        int write_8(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }


        // writes letter Z and returns the index where the cursor is 
        int write_9(int dim, int ex, int pos){
            int pose_index = pos;
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dash(dim,ex,pose_index);
            pose_index = write_dot(dim, ex, pose_index);
            return pose_index;
        }

        cuv::tensor<float,cuv::host_memory_space> get_data(){
            return data;
        }

};


        random_translation::random_translation(int dim, int num_train_examples, int num_test_examples, float thres, int distance, float sigma, int subsample, int max_translation, int max_growing,int min_size, int max_size, int flag):
            m_num_train_example(num_train_examples),
            m_num_test_example(num_test_examples),
            m_dim(dim),
            m_thres(thres),
            m_distance(distance),
            m_sigma(sigma)
        {
            srand ( time(NULL) );


            initialize_data_sets( train_data,  test_data, train_labels, test_labels,  m_num_train_example,  m_num_test_example,  m_dim,  m_thres,  max_size,  min_size,  max_translation,  max_growing, flag);


            if(subsample > 1){
                //creates gaussian filter
                cuv::tensor<float,cuv::host_memory_space> gauss;
                fill_gauss(gauss, m_distance, m_sigma);

                // convolves last dim of both train and test data with the gauss filter
                convolve_last_dim(train_data, gauss);
                convolve_last_dim(test_data, gauss);

                // subsamples each "subsample" element
                subsampling(train_data, subsample);
                subsampling(test_data,subsample);
            }


            normalize_data_set(train_data);
            normalize_data_set(test_data); 
        }


        // creates the vector, which is used to randomly translate/grow each example in the dataset which is being created
        void init_transformations(vector<int>& src, unsigned int num_examples, int max_offset){
            srand ( time(NULL) );
            src = vector<int>(num_examples);
            for(unsigned int i = 0; i < num_examples; i++){
                src[i] = rand() % (2 * max_offset  + 1) - max_offset;
            }
        }

        void initialize_data_sets(cuv::tensor<float,cuv::host_memory_space>& train_data, cuv::tensor<float,cuv::host_memory_space>& test_data, 
                                  cuv::tensor<float,cuv::host_memory_space>& train_labels, cuv::tensor<float,cuv::host_memory_space>& test_labels,
                                  int m_num_train_example, int m_num_test_example, int m_dim, float m_thres, int max_size, int min_size, 
                                  int max_translation, int max_growing, int flag){

            bool translated = max_translation > 0;
            int num_transformations = 2 * max_translation + 1;

            train_data.resize(cuv::extents[3][m_num_train_example][m_dim]);
            test_data.resize(cuv::extents[3][m_num_test_example][m_dim]);
            train_labels.resize(cuv::extents[m_num_train_example][num_transformations]);
            test_labels.resize(cuv::extents[m_num_test_example][num_transformations]);
            
            if (flag == 0){
                // fills the train and test sets with random uniform numbers
                cuv::fill_rnd_uniform(train_data);
                cuv::fill_rnd_uniform(test_data);
                cuv::apply_scalar_functor(train_data,cuv::SF_LT,m_thres);
                cuv::apply_scalar_functor(test_data,cuv::SF_LT,m_thres);
            }else if(flag == 1){
                // initializes the data by randomly writing a single bars with random dimension between min_size and max_size 
                cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[3][m_dim * (max_size - min_size) * (max_translation * 2 + 1)][m_dim]);
                cuv::tensor<float,cuv::host_memory_space> labels(cuv::extents[m_dim * (max_size - min_size) * (max_translation * 2 + 1)][num_transformations]);
                initialize_data_set_iter(max_size, min_size, data, m_dim, max_translation);
                split_data_set(data, labels, train_data, test_data, train_labels, test_labels, m_num_train_example, m_dim);
                translated = false;
            }else{
                // morse code
                cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[3][m_dim * 36 * num_transformations][m_dim]);
                cuv::tensor<float,cuv::host_memory_space> labels(cuv::extents[m_dim * 36 * num_transformations][num_transformations]);
                initialize_morse_code(data, labels, m_dim, max_translation);
                split_data_set(data, labels, train_data, test_data, train_labels, test_labels, m_num_train_example, m_dim);
                translated = false;
            }
          
            vector<int> random_translations_train;
            vector<int> random_translations_test;
            vector<int> random_growing_train;
            vector<int> random_growing_test;
            // creates the vectors for random translation/growing. It is used to randomly translate/grow each example in dataset 
            if(max_translation > 0 && translated){
                init_transformations(random_translations_train, train_data.shape(1), max_translation);
                init_transformations(random_translations_test, test_data.shape(1), max_translation);
            }
            if(max_growing > 0){
                init_transformations(random_growing_train, train_data.shape(1), max_growing);
                init_transformations(random_growing_test, test_data.shape(1), max_growing);
            }
            
            // translate and grow train and test data for the first dimension
            if(translated){ 
               translate_data(train_data, 1, random_translations_train);
               translate_data(test_data, 1, random_translations_test);
            }
            if(max_growing > 0 && translated){
                growing_data(train_data, 1, true, random_growing_train);
                growing_data(test_data, 1, true, random_growing_test);
            }
            else if(max_growing > 0 && !translated){
                growing_data(train_data, 1, false, random_growing_train);
                growing_data(test_data, 1, false, random_growing_test);
            }
            if(translated){
               translate_data(train_data, 2, random_translations_train);
               translate_data(test_data, 2, random_translations_test);
            }
            

            if(flag != 0 && max_growing > 0){
                for (unsigned int i = 0; i < random_growing_train.size();i++){
                    random_growing_train[i] *=2;
                }
                for (unsigned int i = 0; i < random_growing_test.size();i++){
                    random_growing_test[i] *=2;
                }
            }

            // translate and grow train and test data for the second dimension
            if(max_growing > 0 && translated){
                growing_data(train_data, 2, true, random_growing_train);
                growing_data(test_data, 2, true, random_growing_test);
            }
            else if(max_growing > 0 && !translated){
                growing_data(train_data, 2, false, random_growing_train);
                growing_data(test_data, 2, false, random_growing_test);
            }

        }


        // initializes the data in the way that ones are next to each other
        void split_data_set(cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels, cuv::tensor<float,cuv::host_memory_space>& train_set, cuv::tensor<float,cuv::host_memory_space>& test_set, cuv::tensor<float,cuv::host_memory_space>& train_labels, cuv::tensor<float,cuv::host_memory_space>& test_labels, int num_examples, int dim){
            std::cout << " num_examples " << num_examples << " total num " << data.shape(1) << std::endl;
            assert((unsigned int)num_examples * 2 < data.shape(1));
            shuffle(data, labels);
            for(int ex = 0; ex < num_examples * 2; ex+=2){
                for(unsigned int i = 0; i < labels.shape(1); i++){
                    train_labels(ex/2,i) = labels(ex,i);
                    test_labels(ex/2,i) = labels(ex + 1,i);
                }


                for(int d = 0; d < dim; d++){
                    for(int s = 0; s < 3; s++){
                        train_set(s,ex / 2,d) = data(s,ex,d);
                        test_set(s,ex / 2,d) = data(s,ex + 1,d);
                    }
                }
            }
        }
        
        // shuffles the examples in the dataset
        void shuffle(cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels){
            srand ( time(NULL) );
            int r = 0;
            float temp = 0.f;
            for(unsigned int ex = 0; ex < data.shape(1); ex++){
                r = ex + (rand() % (data.shape(1) - ex));

                // shuffle labels
                for(unsigned int i = 0; i < labels.shape(1); i++){
                    temp = labels(ex,i);
                    labels(ex,i) = labels(r,i);
                    labels(r,i) = temp;
                }

                // shuffle data
                for(unsigned int s = 0; s < data.shape(0); s++){
                    for(unsigned int dim = 0; dim < data.shape(2); dim++){
                        temp = data(s,ex,dim);
                        data(s,ex,dim) = data(s,r,dim);
                        data(s,r,dim) = temp;
                    }   

                }   

            }   
        }

        // initializes the data in the way that ones are next to each other
        void initialize_data_set_iter(int max_size, int min_size, cuv::tensor<float,cuv::host_memory_space>& data, int m_dim, int max_trans){
            int example = 0;
            int width = 0;
            int diff = max_size - min_size;
            int index = 0;
            int index_2 = 0;
            for(int w = 0; w < diff; w++){
                width = min_size + w;     // width of the signal
                for(int dim = 0; dim < m_dim; dim++){
                    for(int tran = -max_trans; tran <= max_trans; tran++){
                        for(int d = 0; d < m_dim; d++){

                            index = d + tran;           // index of the translated element
                            // wrap around if the index goes out of border
                            if(index < 0)
                                index = m_dim + index;
                            else if(index >= m_dim)
                                index = index - m_dim;

                            if(dim + width < m_dim){
                                if(d >= dim && d <= dim+width){
                                    data(0,example, d) = 1.f;
                                }
                            }else{
                                if(d >=dim || d <= (dim + width - m_dim)){
                                    data(0,example, d) = 1.f;
                                }
                            }

                            data(1, example, index) = data(0,example, d);  //translating first dimension

                            //translating second dimension
                            index_2 = index + tran;           // index of the translated element
                            // wrap around if the index goes out of border
                            if(index_2 < 0)
                                index_2 = m_dim + index_2;
                            else if(index_2 >= m_dim)
                                index_2 = index_2 - m_dim;
                            data(2, example, index_2) = data(1,example, index);  //translating first dimension

                        }    
                        example++;
                    }
                }

            }
        }
        
        // returns the wrap around index
        int get_wrap_index(int size, int pos){
            if(pos >= size){
                return pos - size;
            }
            else if (pos < 0){
                return size + pos;
            }
            else{
                return pos;
            }
        }

        // initializes the morse code
        void initialize_morse_code(cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels, int m_dim, int max_trans){
            data = 0.f;
            labels = 0.f;
            Morse_code morse(data);
            int example = 0;
            for(int tran = -max_trans; tran <= max_trans; tran++){
                for(int dim = 0; dim < m_dim; dim++){
                    morse.write_a(0, example, dim);
                    morse.write_a(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_a(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_b(0, example, dim);
                    morse.write_b(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_b(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_c(0, example, dim);
                    morse.write_c(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_c(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_d(0, example, dim);
                    morse.write_d(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_d(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_e(0, example, dim);
                    morse.write_e(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_e(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_f(0, example, dim);
                    morse.write_f(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_f(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_g(0, example, dim);
                    morse.write_g(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_g(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_h(0, example, dim);
                    morse.write_h(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_h(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_i(0, example, dim);
                    morse.write_i(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_i(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_j(0, example, dim);
                    morse.write_j(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_j(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_k(0, example, dim);
                    morse.write_k(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_k(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_l(0, example, dim);
                    morse.write_l(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_l(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_m(0, example, dim);
                    morse.write_m(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_m(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_n(0, example, dim);
                    morse.write_n(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_n(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_o(0, example, dim);
                    morse.write_o(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_o(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_p(0, example, dim);
                    morse.write_p(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_p(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_q(0, example, dim);
                    morse.write_q(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_q(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_r(0, example, dim);
                    morse.write_r(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_r(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_s(0, example, dim);
                    morse.write_s(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_s(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_t(0, example, dim);
                    morse.write_t(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_t(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_u(0, example, dim);
                    morse.write_u(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_u(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_v(0, example, dim);
                    morse.write_v(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_v(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_w(0, example, dim);
                    morse.write_w(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_w(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_x(0, example, dim);
                    morse.write_x(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_x(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_y(0, example, dim);
                    morse.write_y(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_y(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_z(0, example, dim);
                    morse.write_z(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_z(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;


                    morse.write_0(0, example, dim);
                    morse.write_0(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_0(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_1(0, example, dim);
                    morse.write_1(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_1(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_2(0, example, dim);
                    morse.write_2(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_2(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_3(0, example, dim);
                    morse.write_3(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_3(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_4(0, example, dim);
                    morse.write_4(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_4(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_5(0, example, dim);
                    morse.write_5(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_5(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_6(0, example, dim);
                    morse.write_6(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_6(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_7(0, example, dim);
                    morse.write_7(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_7(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_8(0, example, dim);
                    morse.write_8(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_8(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;

                    morse.write_9(0, example, dim);
                    morse.write_9(1, example, get_wrap_index(m_dim, dim + tran));
                    morse.write_9(2, example, get_wrap_index(m_dim, dim +  2*tran));
                    labels(example, tran + max_trans) = 1.f;
                    example++;
                }
            }
            data = morse.get_data().copy();
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

        //the second dim is the translated version of the first,
        //and third dimension is translated version of the second
        void growing_data(cuv::tensor<float,cuv::host_memory_space>  &data, int dim, bool translated_before, const vector<int> &rand_growing){
            cuv::tensor<float,cuv::host_memory_space> temp_data = data.copy();
           assert(dim == 1 || dim == 2);
           int not_trans = 0;
           if(!translated_before)
               not_trans = 1;

           int speed;
           for(unsigned int i = 0; i < data.shape(1); i++){
               for(unsigned int j = 0; j < data.shape(2); j++){
                   speed = rand_growing[i];
                   // growing
                   if(speed > 0){
                        if (data(dim - not_trans,i,j) == 0.f){
                            for(int s = - speed; s <= speed; s++){
                                int index = j + s;
                                // wrap around
                                if(index < 0)
                                    index = data.shape(2) + index;
                                else if(index >= (int)data.shape(2))
                                    index = index - data.shape(2);

                                if(data(dim - not_trans,i, index) == 1.f)
                                    temp_data(dim, i, j) = 1;
                            }
                        }
                        else
                            temp_data(dim, i, j) = 1;

                   }
                   //shrinking
                   else if (speed < 0){
                        if(data(dim - not_trans, i, j) == 1.f){
                            temp_data(dim, i ,j) = 1;
                            bool set_to_zero = false;
                            for(int s = speed; s <= -speed; s++){
                                int index = j + s;
                                // wrap around
                                if(index < 0)
                                    index = data.shape(2) + index;
                                else if(index >= (int)data.shape(2))
                                    index = index - data.shape(2);
                                if (data(dim- not_trans, i, index) == 0.f){
                                    set_to_zero = true;
                                }
                            }
                            if(set_to_zero)
                                temp_data(dim, i, j) = 0;
                        }
                   }else
                       temp_data(dim, i, j) = temp_data(dim- not_trans, i ,j);

               }
           }
           data = temp_data.copy();

        }

}


