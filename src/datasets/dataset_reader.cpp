
#include "dataset_reader.hpp"
namespace cuvnet
{

    dataset_reader::dataset_reader(std::string file_name_train, std::string file_name_test, std::string file_name_val):
        m_file_train(file_name_train),
        m_file_test(file_name_test),
        m_file_val(file_name_val)
        {
        }
    
        void dataset_reader::read_from_file(){
            read(train_data, train_labels, m_file_train);
            read(test_data, test_labels, m_file_test);
            if(m_file_val != ""){
                read(val_data, val_labels, m_file_val);
            }
        }

        void dataset_reader::read(tensor_type& data, tensor_type& labels,const std::string file_name){
            using namespace cuv;
            
            std::ifstream readfile(file_name.c_str());
            boost::archive::binary_iarchive oa_read(readfile);

            oa_read >> data;
            oa_read >> labels;

            readfile.close();
        }
}
