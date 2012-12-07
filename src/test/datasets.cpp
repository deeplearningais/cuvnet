#include <vector>
#include <algorithm>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/dataset_dumper.hpp>
#include <datasets/voc_detection.hpp>
#include <datasets/dataset_reader.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

#include <boost/test/unit_test.hpp>
using namespace cuvnet;
using namespace std;


BOOST_AUTO_TEST_SUITE( VOC_Detection )
BOOST_AUTO_TEST_CASE(init){
    // NOTE: must be of same size as the squared-size the dataset produces
    //       since bndbox coordinates will be translated to new size!
    std::string fn = "lena.jpg"; 
    {
        std::ofstream os("test.txt");
        os << fn;
        os << " 2";         // two objects
        os << " 0 0 0 1 2 3";     // class 0, not truncated,  xmin,xmax, ymin, ymax
        os << " 1 0 10 11 12 13"; // class 1, not truncated,  xmin,xmax, ymin, ymax
        os << endl;
    }
    voc_detection_dataset ds("test.txt", "test.txt", 1); // 1 thread only
    // training set is shuffled, switch to test set
    ds.switch_dataset(voc_detection_dataset::SS_TEST);
    while(ds.size_available() < 2);
    std::list<voc_detection_dataset::pattern> L;
    ds.get_batch(L, 2);
    BOOST_FOREACH(voc_detection_dataset::pattern& pat, L){
        BOOST_CHECK_EQUAL(pat.meta_info.filename, fn);
        BOOST_REQUIRE_EQUAL(pat.meta_info.objects.size(), 2);

        // first object
        BOOST_CHECK_EQUAL( 0, pat.meta_info.objects[0].klass);
        BOOST_CHECK_EQUAL( 0, pat.meta_info.objects[0].bb.xmin);
        BOOST_CHECK_EQUAL( 1, pat.meta_info.objects[0].bb.xmax);
        BOOST_CHECK_EQUAL( 2, pat.meta_info.objects[0].bb.ymin);
        BOOST_CHECK_EQUAL( 3, pat.meta_info.objects[0].bb.ymax);
        
        // second object
        BOOST_CHECK_EQUAL( 1, pat.meta_info.objects[1].klass);
        BOOST_CHECK_EQUAL( 10, pat.meta_info.objects[1].bb.xmin);
        BOOST_CHECK_EQUAL( 11, pat.meta_info.objects[1].bb.xmax);
        BOOST_CHECK_EQUAL( 12, pat.meta_info.objects[1].bb.ymin);
        BOOST_CHECK_EQUAL( 13, pat.meta_info.objects[1].bb.ymax);

    }
}

BOOST_AUTO_TEST_CASE(realdata){
    //return;

    const char* realtest = "/home/local/datasets/VOC2011/voc_detection_val.txt";
    voc_detection_dataset ds(realtest, realtest);
    ds.switch_dataset(voc_detection_dataset::SS_TEST);

    while(ds.size_available() < 2);

    std::list<voc_detection_dataset::pattern> L;
    ds.get_batch(L, 2);
    BOOST_FOREACH(voc_detection_dataset::pattern& pat, L){
        cuv::libs::cimg::show(pat.img, "image");
        for (unsigned int i = 0; i < pat.tch.shape(0); ++i)
        {
            cuv::libs::cimg::show(pat.ign[cuv::indices[i][cuv::index_range()][cuv::index_range()]], "ignore");
            cuv::libs::cimg::show(pat.tch[cuv::indices[i][cuv::index_range()][cuv::index_range()]], "teacher");
        }
    }
}



BOOST_AUTO_TEST_CASE(DatasetDumper){


    std::cout << " starting test dumper" << std::endl; 
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
    std::cout << "test dataset dumper finished succesfully" << std::endl;
}



BOOST_AUTO_TEST_SUITE_END()
