#include <cmath>
#include <stdexcept>
#include <cstdio>

#include <boost/format.hpp>
#include <boost/thread.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuvnet/datasets/image_queue.hpp>

#include <boost/test/unit_test.hpp>

using namespace cuvnet;

namespace{
    struct dummy_image_meta_info{
        int id;
    };
    struct dummy_pattern : public dummy_image_meta_info{
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
