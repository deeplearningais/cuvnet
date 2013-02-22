#include <vector>
#include <algorithm>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/dataset_dumper.hpp>
#include <datasets/dataset_reader.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

#include <boost/test/unit_test.hpp>
using namespace cuvnet;
using namespace std;


BOOST_AUTO_TEST_SUITE( t_datasets )

    BOOST_AUTO_TEST_CASE(DatasetDumper){

        std::string file_name1 = "test_dumper_train.dat";
        std::string file_name2 = "test_dumper_test.dat";

        cuv::tensor<float,cuv::host_memory_space> act(cuv::extents[1][3]);
        cuv::tensor<float,cuv::host_memory_space> act2(cuv::extents[1][3]);
        cuv::tensor<float,cuv::host_memory_space> all_act(cuv::extents[2][3]);
        cuv::tensor<float,cuv::host_memory_space> labels(cuv::extents[1][2]);
        cuv::tensor<float,cuv::host_memory_space> labels2(cuv::extents[1][2]);
        cuv::tensor<float,cuv::host_memory_space> all_labels(cuv::extents[2][2]);

        act(0,0) = 0.f;
        act(0,1) = 0.6f;
        act(0,2) = 0.3f;

        labels(0,0) = 1.3f;
        labels(0,1) = 0.53f;

        act2(0,0) = 0.4f;
        act2(0,1) = 1.f;
        act2(0,2) = 0.63f;

        labels2(0,0) = 0.42f;
        labels2(0,1) = 1.1f;

        all_act[cuv::indices[0][cuv::index_range()]] = act;
        all_act[cuv::indices[1][cuv::index_range()]] = act2;
        all_labels[cuv::indices[0][cuv::index_range()]] = labels;
        all_labels[cuv::indices[1][cuv::index_range()]] = labels2;
        {
            dataset_dumper dum(file_name1, 2, 1,  3, 2);
            dum.write_to_file(act,labels);
            dum.write_to_file(act2,labels2);
        }
        {
            dataset_dumper dum2(file_name2, 2, 1, 3, 2);
            dum2.write_to_file(act.copy(),labels.copy());
            dum2.write_to_file(act2.copy(),labels2.copy());
        }
        {
            dataset_reader reader(file_name1, file_name2);
            reader.read_from_file();
            for (unsigned int i = 0; i < reader.train_data.shape(0); ++i)
            {
                for (unsigned int j = 0; j < reader.train_data.shape(1); ++j)
                {
                    BOOST_CHECK_EQUAL(all_act(i,j), reader.train_data(i,j));
                    BOOST_CHECK_EQUAL(all_act(i,j),reader.test_data(i,j));
                }
                for (unsigned int j = 0; j < reader.train_labels.shape(1); ++j)
                {
                    BOOST_CHECK_EQUAL(all_labels(i,j), reader.train_labels(i,j));
                    BOOST_CHECK_EQUAL(all_labels(i,j), reader.test_labels(i,j));
                }
            }
        }
    }



BOOST_AUTO_TEST_SUITE_END()
