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

        
        void write_to_file(const tensor_type& act){
           m_oa_log << act; 
        }

        tensor_type read_from_file(){
            std::ifstream readfile(m_file.c_str());
            boost::archive::binary_iarchive oa_read(readfile);

            using namespace cuv;
            int num_batches;

            oa_read >> num_batches;
            tensor_type temp;
            oa_read >> temp;
            int bs = temp.shape(0);

            // read first tensor and init main tensor
            tensor_type ds(extents[bs * num_batches][temp.shape(1)]);
            ds[indices[index_range(0, bs)][index_range()]] = temp;

            // read one by one tensor 
            for (int i = 1; i < num_batches; ++i)
            {
                oa_read >> temp;
                ds[indices[index_range(bs*i, bs *(i+1))][index_range()]] = temp;
            }
            readfile.close();
            return ds;
        }

        void close(){
            m_logfile.close();
        }
    };

}
#endif /* __DATASET_DUMPER_HPP__ */
