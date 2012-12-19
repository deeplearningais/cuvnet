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
    /**
     * dump a dataset to a file, to be used later by dataset_reader.
     *
     * @ingroup datasets
     */
    class dataset_dumper{
        typedef cuv::tensor<float,cuv::host_memory_space> tensor_type;
        

        private:
        /// file where we write the hidden activations and parameters
        std::string m_file; 
        int m_num_batches;
        int m_bs;
        int m_current_batch_num;
        tensor_type m_whole_data;
        tensor_type m_whole_labels;
        std::string m_log_param;

        public:

        /**
         * constructor.
         * 
         * @param file_name where to save the file
         * @param num_batches the total number of batches to save
         * @param bs the batch size
         * @param data_dim_2 size of second dimension of the data
         * @param label_dim_2 size of second dimension of the labels         
         */
        dataset_dumper(std::string file_name, int num_batches, int bs, int data_dim_2, int label_dim_2, std::string log_param = "");

        /**
         * destructor, closes the files
         *
         */
        ~dataset_dumper();
        
        /**
         * Accumulates the batches in memory and dumps them to a file when complete.
         *
         * @param data the inputs
         * @param labels the corresponding labels
         */
        void write_to_file(const tensor_type& data, const tensor_type& labels);

    };

}
#endif /* __DATASET_DUMPER_HPP__ */
