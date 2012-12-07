#include <cmath>
#include <stdexcept>
#include <cstdio>

#include <boost/format.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <datasets/bounding_box_tools.hpp>

#include <cuvnet/tools/preprocess.hpp>
#include <cuvnet/tools/data_source.hpp>

#include <boost/test/unit_test.hpp>

#include <CImg.h>

using namespace cuvnet;

BOOST_AUTO_TEST_SUITE( t_bbtools )

BOOST_AUTO_TEST_CASE( loadsave ){
    using namespace cuvnet::bbtools;
    image img("bbtools.jpg");

    object o;

    // Bird
    o.klass = 0;
    o.bb.xmin = 156;
    o.bb.ymin = 101;
    o.bb.xmax = o.bb.xmin + 196;
    o.bb.ymax = o.bb.ymin + 119;
    img.meta.objects.push_back(o);

    // Primate
    o.klass = 1;
    o.bb.xmin = 385;
    o.bb.ymin = 45;
    o.bb.xmax = o.bb.xmin + 90;
    o.bb.ymax = o.bb.ymin + 94;
    img.meta.objects.push_back(o);

    // Deer
    o.klass = 2;
    o.bb.xmin = 186;
    o.bb.ymin = 266;
    o.bb.xmax = o.bb.xmin + 119;
    o.bb.ymax = o.bb.ymin + 73;
    img.meta.objects.push_back(o);
    {
        sub_image si(img, o.bb);
        si.crop().show("deer");
    }

    rectangle too_large; // original image is 640x480 transposed
    too_large.xmin = -10; // 10 px to either side
    too_large.xmax =  479; 
    too_large.ymin = -10;
    too_large.ymax =  649;


    img.transpose();

    {
        sub_image si(img, too_large);
        BOOST_CHECK_THROW(si.constrain_to_orig(false), std::runtime_error);
        BOOST_CHECK_NO_THROW(si.constrain_to_orig(true));
        si.crop().show("whole image, no objects marked"); // whole image!?
    }

    {
        sub_image si(img, too_large);
        BOOST_CHECK_THROW(si.constrain_to_orig(false), std::runtime_error);
        BOOST_CHECK_NO_THROW(si.constrain_to_orig(true).constrain_to_orig(true).extend_to_square());
        si.crop_with_padding().mark_objects().show("whole image with padding"); // whole image with padding!?
    }
    {   // mark type
        sub_image si(img, too_large);
        si.constrain_to_orig(true).crop_with_padding();
        si.mark_objects(2,255,0.1);
        si.show("objects marked w/ blobs"); 
    }

    {
        sub_image si(img, img.meta.objects.back().bb);
        si.extend_to_square();
        BOOST_CHECK_EQUAL(si.pos.xmax-si.pos.xmin, si.pos.ymax-si.pos.ymin);
        si.crop().show("squared deer"); 
        si.scale_larger_dim(256).mark_objects().show("squared deer, 256x256"); 
    }

    {   // keeping track of margins (1): setting margins to a fixed value.
        sub_image si(img, too_large);
        si.extend_to_square();
        si.crop_with_padding().scale_larger_dim(172).fill_padding(128).mark_objects().show("whole image with margin set to gray"); 
    }
    {   // keeping track of margins (2): removing margins which are not part of the image
        sub_image si(img, too_large);
        si.extend_to_square();
        si.crop_with_padding().scale_larger_dim(800).fill_padding(128).remove_padding().mark_objects().show("Padding removed"); 
    }
}

BOOST_AUTO_TEST_SUITE_END()


/*
 *BOOST_AUTO_TEST_SUITE( data_source_test )
 *
 *BOOST_AUTO_TEST_CASE(t_folder_loader){
 *    folder_loader fl("/home/local/datasets/VOC2011/TrainVal/VOCdevkit/VOC2011/JPEGImages",false);
 *    filename_processor fp;
 *
 *    typedef cuv::tensor<float,cuv::host_memory_space> tens_t;
 *    std::vector<tens_t> v;
 *    fl.get(v, 16, &fp);
 *    std::vector<unsigned int> shape = v[0].shape();
 *    std::copy(shape.begin(),shape.end(),std::ostream_iterator<unsigned int>(std::cout,", "));
 *    for (int i = 0; i < 16; ++i)
 *    {
 *        using namespace cuv;
 *        tensor<float,cuv::host_memory_space> img = v[i];
 *        libs::cimg::save( img, boost::str(boost::format("pp_img%03d.png")%i));
 *    }
 *}
 *
 *BOOST_AUTO_TEST_SUITE_END()
 */
