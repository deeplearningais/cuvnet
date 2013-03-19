#include "dataset_dumper.hpp"
#include <log4cxx/logger.h>
#include<cuv/basics/io.hpp>
#include<boost/archive/binary_oarchive.hpp>
#include<boost/archive/binary_iarchive.hpp>
#include<cuvnet/tools/matwrite.hpp>
namespace cuvnet
{
     
        dataset_dumper::dataset_dumper(std::string file_name, int num_batches, int bs, int data_dim_2, int label_dim_2, std::string format):
            m_file(file_name),
            m_num_batches(num_batches),
            m_bs(bs),
            m_current_batch_num(0),
            m_format(format)
        {
            m_whole_data.resize(cuv::extents[num_batches * bs][data_dim_2]);
            m_whole_labels.resize(cuv::extents[num_batches * bs][label_dim_2]);
        }
    

        dataset_dumper::~dataset_dumper()
        {
            if(m_current_batch_num < m_num_batches){
                log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("dataset_dumper"));
                LOG4CXX_WARN(log, "Data is not dumped to file because not all batches"
                                  << " are accumulated. Current batch number: " 
                                   << m_current_batch_num << " number of batches: " << m_num_batches);
            }
        }
        void dataset_dumper::accumulate_batch(tensor_type data, tensor_type labels){
           if (m_current_batch_num < m_num_batches){
               if(data.ndim() != 2)
                   data.reshape(cuv::extents[m_bs][m_whole_data.shape(1)]);
               if(labels.ndim() != 2)
                   labels.reshape(cuv::extents[m_bs][m_whole_labels.shape(1)]);

               m_whole_data[cuv::indices[cuv::index_range(m_current_batch_num * m_bs, (m_current_batch_num+1) * m_bs)]] = data;
               m_whole_labels[cuv::indices[cuv::index_range(m_current_batch_num * m_bs,(m_current_batch_num+1) * m_bs)]] = labels;
               m_current_batch_num++;
               //writes to the file the whole data ones it is accumulated 
               if(m_current_batch_num == m_num_batches){
                   if(m_format == "tensors"){
                       std::ofstream logfile (m_file.c_str());
                       boost::archive::binary_oarchive oa_log(logfile);
                       oa_log << m_whole_data; 
                       oa_log << m_whole_labels;
                       logfile.close();
                   }else if(m_format == "numpy"){
                       tofile(m_file + "-data.npy", m_whole_data);
                       tofile(m_file + "-labels.npy", m_whole_labels);
                   }else{
                       throw std::runtime_error("data_dumper: Unknown file format for writing activations/labels to file.");
                   }
               }
           }else{
                log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("dataset_dumper"));
                LOG4CXX_WARN(log, "It is tried to accumulate more batches than the"
                                   << " maximum number of batches: " << m_current_batch_num
                                   << " number of batches: " << m_num_batches);
           }
        }
}
