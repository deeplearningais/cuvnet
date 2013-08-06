#include <cmath>
#include <stdexcept>
#include <cstdio>

#include <boost/format.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <datasets/bounding_box_tools.hpp>
#include <datasets/image_queue.hpp>

#include <cuvnet/tools/preprocess.hpp>
#include <cuvnet/tools/data_source.hpp>

#include <boost/test/unit_test.hpp>

#include <CImg.h>

using namespace cuvnet;
using namespace cuvnet::image_datasets;

struct image_loader_factory{
    output_properties* m_output_properties;

    image_loader_factory(output_properties* op)
        : m_output_properties(op) { }

    whole_image_loader
    operator()(image_queue<pattern>* q, 
                const bbtools::image_meta_info* meta){
        return whole_image_loader(q, meta, m_output_properties, 128, true, 4);
    }
};

BOOST_AUTO_TEST_SUITE( t_bbtools )

BOOST_AUTO_TEST_CASE( image_dataset_test ){

    image_dataset ids("image_dataset.txt", false);
    BOOST_CHECK_EQUAL(2, ids.size());
    BOOST_CHECK_NE(std::string::npos, ids.get(0).filename.find("2008_007573.jpg"));
    BOOST_CHECK_EQUAL(1, ids.get(0).objects.size());
    BOOST_CHECK_EQUAL(2, ids.get(1).objects.size());

}

BOOST_AUTO_TEST_CASE( image_loading_and_queueing ){

    image_dataset ids("image_dataset.txt", false);

    output_properties op(1,1,0,0);
    for (int grayscale = 0; grayscale < 2; ++grayscale)
    {
        image_queue<pattern> q;
        whole_image_loader ld(&q, &ids.get(0), &op, 128, grayscale, 4);

        ld(); // load a single image

        BOOST_CHECK_EQUAL(1, q.size());

        std::list<pattern*> L;
        q.pop(L, 1);

        unsigned int ndims = grayscale ? 1 : 3;
        BOOST_CHECK_EQUAL(ndims, L.front()->img.shape(0));
        BOOST_CHECK_EQUAL(128, L.front()->img.shape(1));
        BOOST_CHECK_EQUAL(128, L.front()->img.shape(2));
    }

    // load a whole bunch of images
    image_queue<pattern> q;
    auto pool = make_loader_pool(2, q, ids, image_loader_factory(&op), 2, 2);

    pool->start();
    boost::this_thread::sleep(boost::posix_time::millisec(500));
    pool->request_stop();

    BOOST_CHECK_LE(2, q.size());
}

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
