#ifndef __DATASET_DUMPER_HPP__
#     define __DATASET_DUMPER_HPP__

#include<iostream>
#include<fstream>
#include<boost/archive/binary_oarchive.hpp>
#include<boost/archive/binary_iarchive.hpp>
#include<cuv.hpp>
#include<cuv/basics/io.hpp>

namespace cuvnet
{
    struct  dataset_reader: public dataset{
        
        typedef cuv::tensor<float,cuv::host_memory_space> tensor_type;
        private:
        std::string m_file_train; 
        std::string m_file_test; 

        public:

        /**
         * constructor
         * 
         */
        dataset_reader(std::string file_name_train, std::string file_name_test):
            m_file_train(file_name_train),
            m_file_test(file_name_test)
        {
        }

        void read_from_file(){
            read(train_data, train_labels, m_file_train);
            read(test_data, test_labels, m_file_test);
        }
        
        // reads data and labels from file
        void read(tensor_type& data, tensor_type& labels,const std::string file_name){
            using namespace cuv;
            std::ifstream readfile(file_name.c_str());
            boost::archive::binary_iarchive oa_read(readfile);

            int num_batches;

            oa_read >> num_batches;
            tensor_type data_batch;
            tensor_type label_batch;
            oa_read >> data_batch;
            oa_read >> label_batch;
            int bs = data_batch.shape(0);

            // read first tensor and init main tensor
            data.resize(extents[bs * num_batches][data_batch.shape(1)]);
            labels.resize(extents[bs * num_batches][label_batch.shape(1)]);
            data[indices[index_range(0, bs)][index_range()]] = data_batch;
            labels[indices[index_range(0, bs)][index_range()]] = label_batch;

            // read one by one tensor 
            for (int i = 1; i < num_batches; ++i)
            {
                oa_read >> data_batch;
                oa_read >> label_batch;
                data[indices[index_range(bs*i, bs *(i+1))][index_range()]] = data_batch;
                labels[indices[index_range(bs*i, bs *(i+1))][index_range()]] = label_batch;
            }
            readfile.close();
        }
    };

    class dataset_dumper{
        typedef cuv::tensor<float,cuv::host_memory_space> tensor_type;

        private:
        /// file where we write the hidden activations and parameters
        std::ofstream m_logfile;


        boost::archive::binary_oarchive m_oa_log;
        std::string m_file; 
        public:

        /**
         * constructor
         * 
         */
        dataset_dumper(std::string file_name, unsigned int num_batches):
            m_logfile(file_name.c_str()),
            m_oa_log(m_logfile),
            m_file(file_name)
        {
            m_oa_log << num_batches;
        }

        
        void write_to_file(const tensor_type& data, const tensor_type& labels){
           m_oa_log << data; 
           m_oa_log << labels;
        }


        void close(){
            m_logfile.close();
        }
    };

}
#endif /* __DATASET_DUMPER_HPP__ */
