#ifndef __DATASET_DUMPER_HPP__
#     define __DATASET_DUMPER_HPP__

#include<iostream>
#include<fstream>
#include<boost/archive/binary_oarchive.hpp>
#include<boost/archive/binary_iarchive.hpp>
#include<cuv.hpp>
#include<cuv/basics/io.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>



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

            oa_read >> data;
            oa_read >> labels;

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
