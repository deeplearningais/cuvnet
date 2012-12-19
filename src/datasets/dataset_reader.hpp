#ifndef __DATASET_READER_HPP__
#     define __DATASET_READER_HPP__

#include<iostream>
#include<fstream>
#include<boost/archive/binary_oarchive.hpp>
#include<boost/archive/binary_iarchive.hpp>
#include<cuv.hpp>
#include<cuv/basics/io.hpp>

#include "dataset.hpp"


namespace cuvnet
{

    /**
     * @ingroup datasets
     *
     * Read a dataset which was serialized using dataset_dumper.
     */
    struct  dataset_reader: public dataset{
        
        typedef cuv::tensor<float,cuv::host_memory_space> tensor_type;
        private:
        std::string m_file_train; 
        std::string m_file_test; 
        std::string m_file_val; 

        public:

        /**
         * constructor.
         * 
         * @param file_name_train the file containing training data
         * @param file_name_test the file containing test data
         * @param file_name_val the file containing test data
         */
        dataset_reader(std::string file_name_train, std::string file_name_test, std::string file_name_val = "");

        /**
         * read the files into memory.
         */
        void read_from_file();
        
        /** 
         * read one file into memory.
         * @param data the tensor to write data to
         * @param labels the tensor to write labels to
         * @param file_name the name of the file to read from
         */
        void read(tensor_type& data, tensor_type& labels,const std::string file_name);
    };

}
#endif /* __DATASET_READER_HPP__ */
