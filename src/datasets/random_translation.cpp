
#include <vector>
#include <cmath>
#include <cuv.hpp>
#include "random_translation.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
namespace cuvnet
{

        // constructor
    morse_code::morse_code(cuv::tensor<float,cuv::host_memory_space> data, int factor):
        m_data(data.copy()),
        m_coordinates(3),
        m_factor(factor)
    {
        init_morse_code_data_structure();
        for (unsigned int i = 0; i < data.shape(0); ++i)
        {
            m_coordinates[i] = std::vector< vector<float> >(data.shape(1));
        } 
    }

        // returns wrap around index
        int morse_code::get_wrap_index(int size, int pos){
            if(pos >= size){
                return pos % size;
            }
            else if(pos % size == 0){
                return 0;
            }
            else if (pos < 0){
                return size + pos % size;
            }
            else{
                return pos;
            }
        }

        std::vector<std::string> morse_code::get_morse_code(){
            return m_morse_code;
        }

        void morse_code::init_morse_code_data_structure(){
            string morse = ".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- "
                ".-- -..- -.-- --.. ----- .---- ..--- ...-- ....- ..... -.... --... ---.. ----. .-.-.- --..--";

            std::istringstream ss(morse, std::istringstream::in);
            while(ss){
                std::string s;
                ss >> s;
                if(s.size())
                    m_morse_code.push_back(s);
            }
            //char letter = 'D';
            //std::cout << v[letter - 'A'];
        }
        int morse_code::char_to_morse_index(char c){
            if(c >= 'A' && c <= 'Z')
                return c-'A';
            if(c >= '0' && c <= '9')
                return char_to_morse_index('Z') + (c-'0');
            if(c == ' ')
                return char_to_morse_index('9') + 1;
            if(c == ',')
                return char_to_morse_index('9') + 2;
            throw std::runtime_error("unrecognized character");
        }
        
        int morse_code::get_width_char(int ch, int factor){
            int width = 0;
            const std::string& str = m_morse_code[ch];
            string::const_iterator it;
            for ( it = str.begin() ; it < str.end(); it++)
            {
                if(*it == '.'){
                    width += 2 * factor;
                }
                else if(*it == '-'){
                    width += 3 * factor;
                }else{
                    throw std::runtime_error("unrecognized character");
                }
            }
            return width - factor;
            
        }

        // writes value 1 at position pos, and 0 after it
        int morse_code::write_dot(int dim, int ex, float pos){
            int new_factor = m_factor;
            int new_pos = pos + new_factor;
            
            // add coordinates of the dot
            m_coordinates[dim][ex].push_back(pos);
            m_coordinates[dim][ex].push_back(new_pos);

            return new_pos + new_factor;
        }

        // writes at position pos 3 times 1 and once 0
        int morse_code::write_dash(int dim, int ex, float pos){
            int new_factor = 2 * m_factor;
            int new_pos = pos +  new_factor;

            // add coordinates of the dot
            m_coordinates[dim][ex].push_back(pos);
            m_coordinates[dim][ex].push_back(new_pos);

            new_factor = m_factor;
            return new_pos + new_factor;
        }
        
        void morse_code::local_translation_speeds(vector<float> &subsampled_pos, float pos_start, float pos_end,  vector<float> &subsampled_speeds, float tran, float scale, int input_size){
            unsigned int size = ceil(pos_end) - floor(pos_start);
            vector<float> orig_coor;
            vector<float> speeds(size);
            subsampled_pos = vector<float>();
            subsampled_speeds = vector<float>();

            // fill posiitons
            for (int i = floor(pos_start); i < ceil(pos_end); ++i)
            {
                orig_coor.push_back(i);
            }
            vector<float> transf_coor(orig_coor);

            // translate coordinates                   
            if (tran != 0.f){
                for (unsigned int c = 0; c < size; c++){
                    transf_coor[c] += tran;
                }
            }

            // scale 
            if (scale !=1.f){
                // mean is the average over the last and first coordinate
                float mean = (transf_coor[size - 1] + transf_coor[0]) / 2;
                //std::cout << " last = " << m_coordinates[dim][ex][size - 1] << " first = " << m_coordinates[dim][ex][0] << " mean " << mean << std::endl;
                for (unsigned int c = 0; c < size; c++){
                    transf_coor[c] -= mean;
                    transf_coor[c] *= scale;
                    transf_coor[c] += mean;
                }
            }

            // estimate local translation speeds
            for (unsigned int c = 0; c < size; c++){
               speeds[c]  = transf_coor[c] - orig_coor[c];

            }

            // wrap around original positions
            for (unsigned int c = 0; c < size; c++){
                orig_coor[c] = get_wrap_index(input_size, orig_coor[c]);
            }
            
            // subsample speeds and positions
            int start_index;
            if((int)orig_coor[0] % 2 == 0){
                start_index = 0;
            }else{
                start_index = 1;
            }
            for (unsigned int c = start_index; c < size; c+=2){
                subsampled_pos.push_back(orig_coor[c] / 2.);
                subsampled_speeds.push_back(speeds[c] / 2.);
            }
            

        }



        void morse_code::local_translation_speeds_all_pos(vector<float> &subsampled_pos,  vector<float> &subsampled_speeds, float tran, float scale, int input_size){
            unsigned int size = input_size;
            vector<float> orig_coor;
            vector<float> speeds(size);
            subsampled_pos = vector<float>();
            subsampled_speeds = vector<float>();

            // fill posiitons
            for (int i = 0; i < input_size; ++i)
            {
                orig_coor.push_back(i);
            }
            vector<float> transf_coor(orig_coor);

            // translate coordinates                   
            if (tran != 0.f){
                for (unsigned int c = 0; c < size; c++){
                    transf_coor[c] += tran;
                }
            }

            // scale 
            if (scale !=1.f){
                // mean is the average over the last and first coordinate
                float mean = (transf_coor[size - 1] + transf_coor[0]) / 2;
                //std::cout << " last = " << m_coordinates[dim][ex][size - 1] << " first = " << m_coordinates[dim][ex][0] << " mean " << mean << std::endl;
                for (unsigned int c = 0; c < size; c++){
                    transf_coor[c] -= mean;
                    transf_coor[c] *= scale;
                    transf_coor[c] += mean;
                }
            }

            // estimate local translation speeds
            for (unsigned int c = 0; c < size; c++){
               speeds[c]  = transf_coor[c] - orig_coor[c];

            }

            // wrap around original positions
            for (unsigned int c = 0; c < size; c++){
                orig_coor[c] = get_wrap_index(input_size, orig_coor[c]);
            }
            
            // subsample speeds and positions
            int start_index;
            if((int)orig_coor[0] % 2 == 0){
                start_index = 0;
            }else{
                start_index = 1;
            }
            for (unsigned int c = start_index; c < size; c+=2){
                subsampled_pos.push_back(orig_coor[c] / 2.);
                subsampled_speeds.push_back(speeds[c] / 2.);
            }
            

        }



        void morse_code::translate_coordinates(int dim, int ex, float trans){
            if(trans != 0.f){
                unsigned int size = m_coordinates[dim][ex].size();
                for (unsigned int c = 0; c < size; c++){
                    m_coordinates[dim][ex][c] += trans;
                }
            }
        }

        // shifts each coordinate by the mean to the left (puts its center at the 0 position) and scales each coordinate and shifts back
        void morse_code::scale_coordinates(int dim, int ex, float scale){
            if (scale != 1.f){
                unsigned int size = m_coordinates[dim][ex].size();
                // mean is the average over the last and first coordinate
                float mean = (m_coordinates[dim][ex][size - 1] + m_coordinates[dim][ex][0]) / 2;
                //std::cout << " last = " << m_coordinates[dim][ex][size - 1] << " first = " << m_coordinates[dim][ex][0] << " mean " << mean << std::endl;
                for (unsigned int c = 0; c < size; c++){
                    m_coordinates[dim][ex][c] -= mean;
                    m_coordinates[dim][ex][c] *= scale;
                    m_coordinates[dim][ex][c] += mean;
                }
            }
        }

        void morse_code::write_from_coordinates(){
           m_data = 0.f;
           for(int d = 0; d < 3; d++){
               for (unsigned int ex = 0; ex < m_data.shape(1); ++ex)
               {
                   unsigned int size = m_coordinates[d][ex].size();
                   for (unsigned int c = 0; c < size; c+= 2)
                   {
                       float c1 = m_coordinates[d][ex][c];
                       float c2 = m_coordinates[d][ex][c+1];
                       //if (c == 0 && d == 1){
                       //std::cout << " c1 " << c1 << " c2 " << c2  << "           c1,last " << m_coordinates[d][ex][size - 2] << " c2,last " << m_coordinates[d][ex][size - 1] << "        c1,0 " << m_coordinates[d-1][ex][c]<< " c2,0 " << m_coordinates[d -1][ex][c+1] << "      c01,last " << m_coordinates[d-1][ex][size - 2] << " c02,last " << m_coordinates[d -1][ex][size - 1]   << "        c1,2 " << m_coordinates[d+1][ex][c]<< " c2,2 " << m_coordinates[d +1][ex][c+1] << "      c1,2,last " << m_coordinates[d+1][ex][size - 2] << " c2,2,last " << m_coordinates[d + 1][ex][size - 1]<< std::endl;
                       //}
                       write_bar(d, ex, c1, c2);
                   }
               }
           } 
        }

        void morse_code::write_bar(int dim, int ex, float coor_1, float coor_2){
            // calculates decimal parts of both coordinates
            int size = m_data.shape(2);
            float dec_part_1 = abs(coor_1 - (int)coor_1); 
            float dec_part_2 = abs(coor_2 - (int)coor_2); 
            float value;
            //if(dim == 0){
            //    std::cout << std::endl;
            //    std::cout << std::endl;
            //    std::cout << " c1 = " << coor_1 << " c2 = " << coor_2 <<  " dec1 " << dec_part_1 << " dec2 " << dec_part_2 << std::endl;
            //}
            for (int c = floor(coor_1); c <= floor(coor_2); ++c)
            {
                
                if(c == floor(coor_1)){                                   // if we are at begining of the bar
                    if(dec_part_1 == 0.f){
                        value = 1.f;
                    }
                    else if(c < 0){
                        value =  dec_part_1; 
                    }else{
                        value = 1 - dec_part_1; 
                    }
                }else if(c == floor(coor_2) && dec_part_2 > 0.f){         // if last coordinate and decimal more than zero than write it there
                    if (c < 0){
                        value = 1 - dec_part_2;
                    }else{
                        value =  dec_part_2;
                    }
                }else if(c == floor(coor_2) && dec_part_2 == 0.f){        // if decimal part is zero, and it is last coordinate then we dont write to that place
                    value = 0.f;
                }else{
                    value = 1.f;
                }
                //if( dim == 0 ){
                //    std::cout << " value " << value << " c " << c  << std::endl;
                //}
                int index = get_wrap_index(size, c);
                m_data(dim, ex, index) += value;
            }
        }

        int morse_code::write_char(int ch, int dim, int ex, float pos){
            const std::string& str = m_morse_code[ch];
            string::const_iterator it;
            for ( it = str.begin() ; it < str.end(); it++)
            {
                if(*it == '.'){
                    pos = write_dot(dim, ex, pos);
                }
                else if(*it == '-'){
                    pos = write_dash(dim, ex, pos);
                }else{
                    throw std::runtime_error("unrecognized character");
                }
            }
            return pos;
        }

        unsigned int morse_code::get_size(){
            return m_morse_code.size();
        }
        cuv::tensor<float,cuv::host_memory_space> morse_code::get_data(){
            return m_data;
        }



        random_translation::random_translation(int dim, int num_train_examples, int num_test_examples, float thres, int distance, float sigma, int subsample, int max_translation, float max_growing,int min_size, int max_size, int flag, int morse_factor):
            m_num_train_example(num_train_examples),
            m_num_test_example(num_test_examples),
            m_dim(dim),
            m_thres(thres),
            m_distance(distance),
            m_sigma(sigma)
        {
            srand ( time(NULL) );


            initialize_data_sets(pattern_ident_train, pattern_ident_test, train_data,  test_data, val_data, train_labels, test_labels,  m_num_train_example,  m_num_test_example,  m_dim,  m_thres,  max_size,  min_size,  max_translation,  max_growing, flag, morse_factor);

            if(subsample > 1){
                //creates gaussian filter
                cuv::tensor<float,cuv::host_memory_space> gauss;
                fill_gauss(gauss, m_distance, m_sigma);

                // convolves last dim of both train and test data with the gauss filter
                convolve_last_dim(train_data, gauss);
                convolve_last_dim(test_data, gauss);
                convolve_last_dim(val_data, gauss);

                // subsamples each "subsample" element
                subsampling(train_data, subsample);
                subsampling(test_data,subsample);
                subsampling(val_data,subsample);
            }


            normalize_data_set(train_data);
            normalize_data_set(test_data); 
            normalize_data_set(val_data); 
        }

       



        // creates the vector, which is used to randomly translate/grow each example in the dataset which is being created
        void init_transformations(vector<int>& src, unsigned int num_examples, int max_offset){
            srand ( time(NULL) );
            src = vector<int>(num_examples);
            for(unsigned int i = 0; i < num_examples; i++){
                src[i] = rand() % (2 * max_offset  + 1) - max_offset;
            }
        }

        void initialize_data_sets(cuv::tensor<float,cuv::host_memory_space>& pattern_ident_train, cuv::tensor<float,cuv::host_memory_space>& pattern_ident_test, cuv::tensor<float,cuv::host_memory_space>& train_data, cuv::tensor<float,cuv::host_memory_space>& test_data, cuv::tensor<float,cuv::host_memory_space>& val_data,
                                  cuv::tensor<float,cuv::host_memory_space>& train_labels, cuv::tensor<float,cuv::host_memory_space>& test_labels,
                                  int m_num_train_example, int m_num_test_example, int m_dim, float m_thres, int max_size, int min_size, 
                                  int max_translation, float max_growing, int flag, int morse_factor){

            bool translated = max_translation > 0;
            //int num_transformations = (2 * max_translation + 1) * (2 * max_growing * 20 + 1);
            int num_transformations = 10;
            int max_num_pos = 100;
            std::cout << " total number of transformation: " << num_transformations << std::endl; 

            int label_dim = m_dim / 2;

            train_data.resize(cuv::extents[3][m_num_train_example][m_dim]);
            val_data.resize(cuv::extents[3][m_num_train_example][m_dim]);
            test_data.resize(cuv::extents[3][m_num_test_example][m_dim]);
            train_labels.resize(cuv::extents[m_num_train_example][label_dim]);
            test_labels.resize(cuv::extents[m_num_test_example][label_dim]);

            pattern_ident_train.resize(cuv::extents[m_num_train_example][label_dim]);
            pattern_ident_test.resize(cuv::extents[m_num_test_example][label_dim]);
            
            if (flag == 0){
                // fills the train and test sets with random uniform numbers
                cuv::fill_rnd_uniform(train_data);
                cuv::fill_rnd_uniform(val_data);
                cuv::fill_rnd_uniform(test_data);
                cuv::apply_scalar_functor(train_data,cuv::SF_LT,m_thres);
                cuv::apply_scalar_functor(val_data,cuv::SF_LT,m_thres);
                cuv::apply_scalar_functor(test_data,cuv::SF_LT,m_thres);
            }else if(flag == 1){
                // initializes the data by randomly writing a single bars with random dimension between min_size and max_size 
                cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[3][m_dim * (max_size - min_size) * (max_translation * 2 + 1)][m_dim]);
                cuv::tensor<float,cuv::host_memory_space> labels(cuv::extents[m_dim * (max_size - min_size) * (max_translation * 2 + 1)][num_transformations]);
                initialize_data_set_iter(max_size, min_size, data, m_dim, max_translation);
                //split_data_set(data, labels, train_data, test_data, val_data, train_labels, test_labels, m_num_train_example, m_dim);
                translated = false;
            }else{
                // morse code
                cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[3][max_num_pos * 38 * num_transformations][m_dim]);
                cuv::tensor<float,cuv::host_memory_space> labels(cuv::extents[max_num_pos * 38 * num_transformations][label_dim]);
                cuv::tensor<float,cuv::host_memory_space> pattern_ident(cuv::extents[max_num_pos * 38 * num_transformations][label_dim]);
                initialize_morse_code(pattern_ident, data, labels, m_dim, max_translation, morse_factor, max_growing);
                split_data_set(pattern_ident, pattern_ident_train, pattern_ident_test, data, labels, train_data, test_data, val_data, train_labels, test_labels, m_num_train_example, m_dim);
                translated = false;
            }
          
            if(flag !=2){
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
                if(max_growing > 0 && !translated){
                    growing_data(train_data, 1, true, random_growing_train);
                    growing_data(test_data, 1, true, random_growing_test);
                }
                else if(max_growing > 0 && translated){
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
                if(max_growing > 0 && !translated){
                    growing_data(train_data, 2, true, random_growing_train);
                    growing_data(test_data, 2, true, random_growing_test);
                }
                else if(max_growing > 0 && translated){
                    growing_data(train_data, 2, false, random_growing_train);
                    growing_data(test_data, 2, false, random_growing_test);
                }
            }

        }


        // initializes the data in the way that ones are next to each other
        void split_data_set(cuv::tensor<float,cuv::host_memory_space>& pattern_ident, cuv::tensor<float,cuv::host_memory_space>& pattern_ident_train, cuv::tensor<float,cuv::host_memory_space>& pattern_ident_test, cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels, cuv::tensor<float,cuv::host_memory_space>& train_set, cuv::tensor<float,cuv::host_memory_space>& test_set, cuv::tensor<float,cuv::host_memory_space>& val_set,
                cuv::tensor<float,cuv::host_memory_space>& train_labels, cuv::tensor<float,cuv::host_memory_space>& test_labels, int num_examples, int dim){

            shuffle(pattern_ident, data, labels);
            std::cout << " num_examples " << num_examples << " total num " << data.shape(1) << std::endl;
            for(int ex = 0; ex < num_examples * 3; ex+=3){
                train_labels[cuv::indices[ex/3][cuv::index_range()]] = labels[cuv::indices[ex][cuv::index_range()]];
                test_labels[cuv::indices[ex/3][cuv::index_range()]] = labels[cuv::indices[ex + 1][cuv::index_range()]];

                pattern_ident_train[cuv::indices[ex/3][cuv::index_range()]] = pattern_ident[cuv::indices[ex][cuv::index_range()]];
                pattern_ident_test[cuv::indices[ex/3][cuv::index_range()]] = pattern_ident[cuv::indices[ex + 1][cuv::index_range()]];
                for(int d = 0; d < 3; d++){
                    train_set[cuv::indices[d][ex/3][cuv::index_range()]] = data[cuv::indices[d][ex][cuv::index_range()]];
                    test_set[cuv::indices[d][ex/3][cuv::index_range()]] = data[cuv::indices[d][ex + 1][cuv::index_range()]];
                    val_set[cuv::indices[d][ex/3][cuv::index_range()]] = data[cuv::indices[d][ex + 2][cuv::index_range()]];
                }

            }
        }
        
        // shuffles the examples in the dataset
        void shuffle(cuv::tensor<float,cuv::host_memory_space>& pattern_ident, cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels){
            std::cout << "shuffling all examples" << std::endl;
            srand ( time(NULL) );
            int r = 0;
            //float temp = 0.f;
            cuv::tensor<float,cuv::host_memory_space> temp_labels(cuv::extents[labels.shape(1)]);
            cuv::tensor<float,cuv::host_memory_space> temp_pattern_ident(cuv::extents[labels.shape(1)]);
            cuv::tensor<float,cuv::host_memory_space> temp_data(cuv::extents[data.shape(2)]);
            for(unsigned int ex = 0; ex < data.shape(1); ex++){
                r = ex + (rand() % (data.shape(1) - ex));

                // shuffle labels
                temp_labels[cuv::indices[cuv::index_range()]] = labels[cuv::indices[ex][cuv::index_range()]]; 
                labels[cuv::indices[ex][cuv::index_range()]] = labels[cuv::indices[r][cuv::index_range()]];
                labels[cuv::indices[r][cuv::index_range()]] =  temp_labels[cuv::indices[cuv::index_range()]];
                
                temp_pattern_ident[cuv::indices[cuv::index_range()]] = pattern_ident[cuv::indices[ex][cuv::index_range()]]; 
                pattern_ident[cuv::indices[ex][cuv::index_range()]] = pattern_ident[cuv::indices[r][cuv::index_range()]];
                pattern_ident[cuv::indices[r][cuv::index_range()]] =  temp_pattern_ident[cuv::indices[cuv::index_range()]];

                // shuffle data
                for(int d = 0; d < 3; d++){
                    temp_data[cuv::indices[cuv::index_range()]] = data[cuv::indices[d][ex][cuv::index_range()]]; 
                    data[cuv::indices[d][ex][cuv::index_range()]] = data[cuv::indices[d][r][cuv::index_range()]];
                    data[cuv::indices[d][r][cuv::index_range()]] = temp_data[cuv::indices[cuv::index_range()]];
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
                return pos % size;
            }
            else if (pos < 0){
                return size + pos % size;
            }
            else{
                return pos;
            }
        }
        // initializes the morse code
        void initialize_morse_code(cuv::tensor<float,cuv::host_memory_space>& pattern_ident, cuv::tensor<float,cuv::host_memory_space>& data, cuv::tensor<float,cuv::host_memory_space>& labels, int m_dim, int max_trans, int morse_factor, float max_grow){
            std::cout << "initializing morse code with max trans " << max_trans << " and max scale " << max_grow << std::endl;
            data = 0.f;
            labels = 0.f;
            pattern_ident = 0.f;
            morse_code morse(data, morse_factor);
            int example = 0;

            srand ( time(NULL) );
            float new_dim;
            int max_num_transf = 10;
            int max_num_pos = 100; 

            for(int t = 0; t < max_num_transf; t++){
                for(int dim = 0; dim < max_num_pos; dim++){
                    for(unsigned int ch = 0; ch < morse.get_size(); ch++){
                        float tran = drand48() * 2*max_trans - max_trans;
                        float rand_grow = (drand48() * 2 * max_grow - max_grow);
                        float grow = 1 + rand_grow;
                        new_dim =  drand48() * m_dim;

                        // generate 1st input
                        float end_pos = morse.write_char(ch, 0, example, new_dim);

                        // generate 2nd input
                        morse.write_char(ch, 1, example,new_dim);
                        if(tran != 0){
                            morse.translate_coordinates(1, example, tran);
                        }
                        if(grow != 1){
                            morse.scale_coordinates(1,example, grow);
                        }

                        // generate teacher
                        morse.write_char(ch, 2, example, new_dim);
                        if(tran != 0){
                            morse.translate_coordinates(2, example, tran);
                            morse.translate_coordinates(2, example, tran);
                        }
                        if(grow != 1){
                            morse.scale_coordinates(2,example, grow);
                            morse.scale_coordinates(2,example, grow);
                        }

                        // fill labels, each pixel is one teacher having the local translation speed as value
                        vector<float> subsampled_pos;
                        vector<float> subsampled_speeds;
                        morse.local_translation_speeds(subsampled_pos, new_dim, end_pos,  subsampled_speeds, tran, grow, m_dim);
                        
                        int start = subsampled_pos.front();
                        int end = subsampled_pos.back();
                        for (int i = start; i <= end; i++){
                            labels(example, i) = subsampled_speeds[i - start];
                            pattern_ident(example, i) = 1;
                        }

                        //labels(example, 0) = tran;
                        //labels(example, 1) = rand_grow * 10;
                        example++;
                    }
                }
            }

            morse.write_from_coordinates();
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


