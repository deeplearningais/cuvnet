#include <cmath>
#include <stdexcept>
#include <cstdio>

#include <boost/format.hpp>
#include <boost/thread.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuvnet/datasets/image_queue.hpp>
#include <cuvnet/datasets/cv_datasets.hpp>

#include <boost/test/unit_test.hpp>

using namespace cuvnet;

namespace{
    struct dummy_image_meta_info{
        int id;
    };
    struct dummy_pattern : public dummy_image_meta_info{
        boost::shared_ptr<pattern_set<dummy_pattern> > set;
        dummy_pattern(dummy_image_meta_info other)
        {
            id = other.id;
            boost::this_thread::sleep_for(boost::chrono::milliseconds((int)(100 * drand48())));
        }
    };
    typedef pattern_set<dummy_pattern> dummy_patternset;

    struct dummy_dataset{
        typedef dummy_patternset patternset_type;
        std::vector<dummy_image_meta_info> meta;
        boost::mutex m_mutex;
        dummy_dataset(): meta(5){
            for(unsigned int i=0;i<meta.size();i++)
                meta[i].id = i;
        }
        size_t size(){return meta.size();}
        boost::shared_ptr<pattern_set<dummy_pattern> > next(size_t offset){
            boost::shared_ptr<pattern_set<dummy_pattern> > ps = 
                boost::make_shared<pattern_set<dummy_pattern> >();
            ps->push(boost::make_shared<dummy_pattern>(meta[offset % meta.size()]));
            return ps;
        }
    };
}

BOOST_AUTO_TEST_SUITE( t_rgb_cls_dataset )
    BOOST_AUTO_TEST_CASE( image_dataset_test ){
        int n_crops = 3;
        int crop_size = 50;
        datasets::rgb_classification_dataset ids("image_dataset.txt", crop_size, n_crops);
        BOOST_REQUIRE_EQUAL(2, ids.size());
        BOOST_CHECK_NE(std::string::npos, ids.m_meta[0].rgb_filename.find("2008_007573.jpg"));
        BOOST_CHECK_NE(std::string::npos, ids.m_meta[1].rgb_filename.find("2008_007576.jpg"));
        BOOST_CHECK_EQUAL(1, ids.m_meta[0].klass);
        BOOST_CHECK_EQUAL(2, ids.m_meta[1].klass);

        for(unsigned int imgidx = 0; imgidx < 2; imgidx++){
            auto patset = ids.next(imgidx);
            BOOST_REQUIRE_EQUAL(patset->todo_size(), n_crops);
            for(int i=0; i<n_crops; i++){
                // first member variable in rgb_image is an size_t.
                BOOST_CHECK_EQUAL(*reinterpret_cast<size_t*>(patset->m_todo[i]->original.get()), imgidx);
                BOOST_CHECK_EQUAL(patset->m_todo[i]->ground_truth_class, imgidx+1);
                BOOST_CHECK_EQUAL(patset->m_todo[i]->rgb.shape(0), 3);
                BOOST_CHECK_EQUAL(patset->m_todo[i]->rgb.shape(1), crop_size);
                BOOST_CHECK_EQUAL(patset->m_todo[i]->rgb.shape(2), crop_size);
            }
        }
    }

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( t_image_queue )

BOOST_AUTO_TEST_CASE( dummy_queue ){
    ThreadPool pool(5);
    for(unsigned int i=0; i<100; i++){
        dummy_dataset ds;

        boost::shared_ptr<dummy_pattern> pat;

        image_queue<dummy_dataset> iq(pool, ds, 2, 4);
        pat = iq.pop();

        BOOST_CHECK_EQUAL(pat->id, 0);
        BOOST_CHECK_EQUAL(iq.m_queue.size(), 3);

        pat = iq.pop();
        BOOST_CHECK_EQUAL(pat->id, 1);
        BOOST_CHECK_EQUAL(iq.m_queue.size(), 2);

        pat = iq.pop();
        BOOST_CHECK_EQUAL(pat->id, 2);
        BOOST_CHECK_EQUAL(iq.m_queue.size(), 1);

        pat = iq.pop();
        BOOST_CHECK_EQUAL(pat->id, 3);
        BOOST_CHECK_EQUAL(iq.m_queue.size(), 3);

        pat = iq.pop();
        BOOST_CHECK_EQUAL(pat->id, 4);
        BOOST_CHECK_EQUAL(iq.m_queue.size(), 2);

        pat = iq.pop();
        BOOST_CHECK_EQUAL(pat->id, 0);
        BOOST_CHECK_EQUAL(iq.m_queue.size(), 1);
    }
}


BOOST_AUTO_TEST_SUITE_END()
