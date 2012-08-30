#include <boost/test/unit_test.hpp>
#include <tools/network_communication.hpp>

#define V(X) #X<<":"<<(X)<<", "
#define HOST "131.220.7.92"
#define DB   "testnc"
#define KEY  "tst"

BOOST_AUTO_TEST_SUITE( netcom )
BOOST_AUTO_TEST_CASE( server_start ){
   using namespace cuvnet;
   using namespace network_communication;
   server s(HOST,DB,KEY);
   s.cleanup();
}
BOOST_AUTO_TEST_CASE( client_start ){
   using namespace cuvnet;
   using namespace network_communication;
   client c(HOST,DB,KEY);
}

BOOST_AUTO_TEST_CASE( save_mat ){
    using namespace cuvnet;
    using namespace network_communication;

    client c(HOST,DB,KEY);
    server s(HOST,DB,KEY);
    s.cleanup();

    cuv::tensor<float, cuv::host_memory_space> n, m(cuv::extents[5][6]);
    cuv::sequence(m);
    c.put_for_merging("m", m);

    BOOST_CHECK_THROW(c.fetch_merged("m"), value_not_found_exception);

    s.pull_merged();
    s.merge();
    s.push_merged();

    BOOST_CHECK_NO_THROW(c.fetch_merged("m"));
    n = c.fetch_merged("m");
    
    BOOST_CHECK_SMALL(cuv::norm1(m-n), 0.001f);
}

BOOST_AUTO_TEST_CASE( averaging ){
    using namespace cuvnet;
    using namespace network_communication;

    client c0(HOST,DB,KEY, "c0");
    client c1(HOST,DB,KEY, "c1");
    server s(HOST,DB,KEY);
    s.cleanup();

    cuv::tensor<float, cuv::host_memory_space> m(cuv::extents[5][6]);
    m = 2.f;
    c0.put_for_merging("m", m);

    m = 3.f;

    c1.put_for_merging("m", m);

    BOOST_CHECK_THROW(c0.fetch_merged("m"), value_not_found_exception);

    s.merge();
    s.push_merged();

    m = c0.fetch_merged("m") - 2.5f;

    BOOST_CHECK_SMALL(cuv::norm1(m), 0.001f);
}

BOOST_AUTO_TEST_CASE( self_overwrite ){
    // a client writes twice before averaging: only last written result should count
    using namespace cuvnet;
    using namespace network_communication;

    client c0(HOST,DB,KEY);
    server s(HOST,DB,KEY);
    s.cleanup();

    cuv::tensor<float, cuv::host_memory_space> m(cuv::extents[5][6]);
    m = 2.f;
    c0.put_for_merging("m", m);

    m = 3.f;

    c0.put_for_merging("m", m);

    BOOST_CHECK_THROW(c0.fetch_merged("m"), value_not_found_exception);

    s.merge();
    s.push_merged();

    m = c0.fetch_merged("m") - 3.0f;

    BOOST_CHECK_SMALL(cuv::norm1(m), 0.001f);
}
BOOST_AUTO_TEST_SUITE_END()
