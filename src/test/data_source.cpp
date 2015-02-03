#include <cmath>
#include <stdexcept>
#include <cstdio>

#include <boost/format.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuvnet/datasets/detection.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE( t_detection )

BOOST_AUTO_TEST_CASE( loading ){
    using namespace datasets;

    rgb_detection_dataset ids("image_dataset.txt", 224, 1);
    BOOST_REQUIRE_EQUAL(2, ids.m_meta.size());
    BOOST_CHECK_NE(std::string::npos, ids.m_meta[0].rgb_filename.find("2008_007573.jpg"));
    BOOST_CHECK_NE(std::string::npos, ids.m_meta[1].rgb_filename.find("2008_007576.jpg"));
    BOOST_CHECK_EQUAL(1, ids.m_meta[0].bboxes.size());
    BOOST_CHECK_EQUAL(2, ids.m_meta[1].bboxes.size());

    boost::shared_ptr<rgb_detection_dataset::patternset_t> patset0
        = ids.next(0);
    boost::shared_ptr<rgb_detection_dataset::patternset_t> patset1
        = ids.next(1);
    
    meta_data<rgb_detection_tag>::show("img0", *patset0->m_todo[0]);
    meta_data<rgb_detection_tag>::show("img1", *patset1->m_todo[0]);
}

BOOST_AUTO_TEST_SUITE_END()
