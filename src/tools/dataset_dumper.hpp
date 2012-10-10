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
    class dataset_dumper{
        typedef cuv::tensor<float,cuv::host_memory_space> tensor_type;
        

        private:
        /// file where we write the hidden activations and parameters
        std::ofstream m_logfile;
        boost::archive::binary_oarchive m_oa_log;
        std::string m_file; 
        int m_num_batches;
        int m_bs;
        int m_data_dim_2;
        int m_label_dim_2;
        int m_current_batch_num;
        tensor_type m_whole_data;
        tensor_type m_whole_labels;

        public:

        /**
         * constructor
         * 
         */
        dataset_dumper(std::string file_name, int num_batches, int bs, int data_dim_2, int label_dim_2):
            m_logfile(file_name.c_str()),
            m_oa_log(m_logfile),
            m_file(file_name),
            m_bs(bs),
            m_data_dim_2(data_dim_2),
            m_label_dim_2(label_dim_2),
            m_current_batch_num(0),
            m_whole_data(cuv::extents[num_batches * bs][data_dim_2]),
            m_whole_labels(cuv::extents[num_batches * bs][label_dim_2])
        {
            
        }

        
        void write_to_file(const tensor_type& data, const tensor_type& labels){
           //m_whole_data[cuv::indices[cuv::index_range(m_current_batch_num * m_bs, (m_current_batch_num+1) * m_bs)][cuv::index_range()]]= data;
           //m_whole_labels[cuv::indices[cuv::index_range(m_current_batch_num * m_bs,(m_current_batch_num+1) * m_bs)][cuv::index_range()]]= labels;
           m_current_batch_num++;
           std::cout << "batch num " << m_current_batch_num << std::endl;/* cursor */
           // writes to the file the whole data ones it is accumulated 
           //if(m_current_batch_num == m_num_batches){
           //    m_oa_log << m_whole_data; 
           //    m_oa_log << m_whole_labels;
           //}
        }


        void close(){
            m_logfile.close();
        }
    };

}
#endif /* __DATASET_DUMPER_HPP__ */
