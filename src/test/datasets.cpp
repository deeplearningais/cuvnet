#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

#include <cuvnet/op_utils.hpp>
#include <datasets/random_translation.hpp>
#include <datasets/voc_detection.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

using namespace cuvnet;
using namespace std;



TEST(RandomTranslation, FillGauss){
    int distance = 3;
    int sigma = 2;
    cuv::tensor<float,cuv::host_memory_space> gauss;
    cuv::tensor<float,cuv::host_memory_space> my_gauss(cuv::extents[2 * distance + 1]);
    fill_gauss(gauss, distance, sigma);
    my_gauss(0) = 0.105399224561864330;
    my_gauss(1) = 0.36787944117144233;
    my_gauss(2) = 0.77880078307140488;
    my_gauss(3) = 1;
    my_gauss(4) = my_gauss(2) ;
    my_gauss(5) = my_gauss(1);
    my_gauss(6) = my_gauss(0);

    float sum = cuv::sum(my_gauss);
    my_gauss /= sum;

    for(int i = 0; i < (int)gauss.shape(0);i++){
        EXPECT_NEAR(my_gauss(i), gauss(i), 0.0001);
    }
}


TEST(RandomTranslation, Convolution){
    cuv::tensor<float,cuv::host_memory_space> simple_filter(cuv::extents[3]);
    simple_filter(0) = 1;
    simple_filter(1) = 1;
    simple_filter(2) = 1;

    cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[1][1][3]);
    data(0,0,0) = 1;
    data(0,0,1) = 2;
    data(0,0,2) = 3;
    convolve_last_dim(data, simple_filter);
    std::cout << "mean : "<< cuv::mean(data)<<std::endl;
    std::cout << "var : "<< cuv::var(data)<<std::endl;
    cuv::tensor<float,cuv::host_memory_space> conv_data(cuv::extents[1][1][3]);
    conv_data(0,0,0) = 6;
    conv_data(0,0,1) = 6;
    conv_data(0,0,2) = 6;
    for(int i = 0; i < (int)conv_data.size(); i++){
        EXPECT_NEAR(conv_data(0,0,i), data(0,0,i), 0.0001);
    }

    data(0,0,0) = 1;
    data(0,0,1) = 2;
    data(0,0,2) = 3;
    simple_filter(0) = 1;
    simple_filter(1) = 2;
    simple_filter(2) = 1;
    convolve_last_dim(data, simple_filter);

    conv_data(0,0,0) = 7;
    conv_data(0,0,1) = 8;
    conv_data(0,0,2) = 9;
    for(int i = 0; i < (int)conv_data.size(); i++){
        EXPECT_NEAR(conv_data(0, 0, i), data(0, 0, i), 0.0001);
    }
}


TEST(RandomTranslation, SubSampling){
    int vec_size = 30;
    cuv::tensor<float,cuv::host_memory_space> tmp_data(cuv::extents[1][1][vec_size]);
    
    int each_elem = 3;
    // fills the data with ones, and each third element with 3
    for(int i = 0; i < (int)tmp_data.shape(0); i++){
        for(int j = 0; j < (int)tmp_data.shape(1); j++){
            for(int k = 0; k < (int)tmp_data.shape(2); k++){
                tmp_data(i,j,k) = 1.f;
                if(j % each_elem == 0)
                    tmp_data(i,j,k) = 0.f;
            }
        }
    }

    subsampling(tmp_data, each_elem);
    int num = cuv::count(tmp_data,0.f);
    EXPECT_EQ(num , vec_size / each_elem);
}


TEST(RandomTranslation, TranslateData){
    cuv::tensor<float,cuv::host_memory_space> tmp_data(cuv::extents[3][1][6]);
    vector<int> rand_trans(tmp_data.shape(1));
    rand_trans[0] = 2;
    tmp_data(0, 0, 0) = 1;    
    tmp_data(0, 0, 1) = 2;    
    tmp_data(0, 0, 2) = 3;    
    tmp_data(0, 0, 3) = 4;    
    tmp_data(0, 0, 4) = 5;    
    tmp_data(0, 0, 5) = 6;

    // translate data by 2    
    translate_data(tmp_data, 1, rand_trans);

    EXPECT_EQ(tmp_data(1,0,0), 5);
    EXPECT_EQ(tmp_data(1,0,1), 6);
    EXPECT_EQ(tmp_data(1,0,2), 1);
    EXPECT_EQ(tmp_data(1,0,3), 2);
    EXPECT_EQ(tmp_data(1,0,4), 3);
    EXPECT_EQ(tmp_data(1,0,5), 4);

    translate_data(tmp_data, 2, rand_trans);
    
    EXPECT_EQ(tmp_data(2,0,0), 3);
    EXPECT_EQ(tmp_data(2,0,1), 4);
    EXPECT_EQ(tmp_data(2,0,2), 5);
    EXPECT_EQ(tmp_data(2,0,3), 6);
    EXPECT_EQ(tmp_data(2,0,4), 1);
    EXPECT_EQ(tmp_data(2,0,5), 2);
    
    // translate data by -2    
    rand_trans[0] = -2;
    translate_data(tmp_data, 1, rand_trans);

    EXPECT_EQ(tmp_data(1,0,0), 3);
    EXPECT_EQ(tmp_data(1,0,1), 4);
    EXPECT_EQ(tmp_data(1,0,2), 5);
    EXPECT_EQ(tmp_data(1,0,3), 6);
    EXPECT_EQ(tmp_data(1,0,4), 1);
    EXPECT_EQ(tmp_data(1,0,5), 2);

    translate_data(tmp_data, 2, rand_trans);
    
    EXPECT_EQ(tmp_data(2,0,0), 5);
    EXPECT_EQ(tmp_data(2,0,1), 6);
    EXPECT_EQ(tmp_data(2,0,2), 1);
    EXPECT_EQ(tmp_data(2,0,3), 2);
    EXPECT_EQ(tmp_data(2,0,4), 3);
    EXPECT_EQ(tmp_data(2,0,5), 4);
}

TEST(VOC_Detection, init){
    // NOTE: must be of same size as the squared-size the dataset produces
    //       since bndbox coordinates will be translated to new size!
    std::string fn = "../src/datasets/util/lena.jpg"; 
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
        EXPECT_EQ(pat.meta_info.filename, fn);
        ASSERT_EQ(pat.meta_info.objects.size(), 2);

        // first object
        EXPECT_EQ( 0, pat.meta_info.objects[0].klass);
        EXPECT_EQ( 0, pat.meta_info.objects[0].xmin);
        EXPECT_EQ( 1, pat.meta_info.objects[0].xmax);
        EXPECT_EQ( 2, pat.meta_info.objects[0].ymin);
        EXPECT_EQ( 3, pat.meta_info.objects[0].ymax);
        
        // second object
        EXPECT_EQ( 1, pat.meta_info.objects[1].klass);
        EXPECT_EQ( 10, pat.meta_info.objects[1].xmin);
        EXPECT_EQ( 11, pat.meta_info.objects[1].xmax);
        EXPECT_EQ( 12, pat.meta_info.objects[1].ymin);
        EXPECT_EQ( 13, pat.meta_info.objects[1].ymax);

    }
}

TEST(VOC_Detection, realdata){
    return;

    const char* realtest = "/home/local/datasets/VOC2011/voc_detection_val.txt";
    voc_detection_dataset ds(realtest, realtest);

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
