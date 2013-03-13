#ifndef __DATASET_DUMPER_HPP__
#     define __DATASET_DUMPER_HPP__

#include<iostream>
#include<fstream>
#include<cuv.hpp>


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
        std::string m_format;

        public:

        /**
         * constructor.
         * 
         * @param file_name where to save the file
         * @param num_batches the total number of batches to save
         * @param bs the batch size
         * @param data_dim_2 size of second dimension of the data
         * @param label_dim_2 size of second dimension of the labels         
         * @param format either "tensors" or "numpy". "tensors" uses the serialization methods of class tensor and saves in ONE file, "numpy" writes two numpy files.
         */
        dataset_dumper(std::string file_name, int num_batches, int bs, int data_dim_2, int label_dim_2, std::string m_format = "numpy");

        /**
         * destructor, closes the files.
         */
        ~dataset_dumper();
        
        /**
         * Accumulates the batches in memory and dumps them to a file when complete.
         *
         * @note that parameters are passed as copies, which allows us to
         * reshape them at little expense.
         *
         * @param data the inputs
         * @param labels the corresponding labels
         */
        void accumulate_batch(tensor_type data, tensor_type labels);

    };

}
#endif /* __DATASET_DUMPER_HPP__ */
