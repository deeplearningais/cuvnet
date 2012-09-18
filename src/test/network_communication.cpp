#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <cuvnet/ops.hpp>
#include <tools/network_communication.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/function.hpp>

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

    BOOST_CHECK_THROW(c.fetch_merged("m"), value_not_found_exception);

    cuv::tensor<float, cuv::host_memory_space> n, m(cuv::extents[5][6]), md;

    cuv::sequence(m);
    md = m.copy();
    md *= 0.1f;

    c.put_for_merging("m", md, m); // only puts m
    c.put_for_merging("m", md, m); // only puts md

    s.merge();
    s.push_merged();

    BOOST_CHECK_NO_THROW(c.fetch_merged("m"));
    n = c.fetch_merged("m");
    
    BOOST_CHECK_SMALL(cuv::norm1((m+md)-n), 0.001f);
}

BOOST_AUTO_TEST_CASE( averaging ){
    using namespace cuvnet;
    using namespace network_communication;

    client c0(HOST,DB,KEY, "c0");
    client c1(HOST,DB,KEY, "c1");
    server s(HOST,DB,KEY);
    s.cleanup();

    BOOST_CHECK_THROW(c0.fetch_merged("m"), value_not_found_exception);
    BOOST_CHECK_THROW(c1.fetch_merged("m"), value_not_found_exception);

    cuv::tensor<float, cuv::host_memory_space> m(cuv::extents[5][6]), md;
    m = 3.f;
    md = m.copy();
    md = 2.f;

    c0.put_for_merging("m", md, m); // puts m
    c0.put_for_merging("m", md, m); // puts md

    md = 4.f;

    c1.put_for_merging("m", md, m); // puts md

    s.merge();
    s.push_merged();

    m = c0.fetch_merged("m") - 9.f;

    BOOST_CHECK_SMALL(cuv::norm1(m), 0.001f);
}

BOOST_AUTO_TEST_CASE( self_overwrite ){
    // a client writes twice before averaging: only last written result should count
    using namespace cuvnet;
    using namespace network_communication;

    client c0(HOST,DB,KEY);
    server s(HOST,DB,KEY);
    s.cleanup();

    BOOST_CHECK_THROW(c0.fetch_merged("m"), value_not_found_exception);

    cuv::tensor<float, cuv::host_memory_space> m(cuv::extents[5][6]), md;
    m = 2.f;
    md = m.copy();
    md = 3.f;

    c0.put_for_merging("m", md, m); // puts m
    c0.put_for_merging("m", md, m); // puts md

    md = 4.f;

    c0.put_for_merging("m", md, m); // overwrites last md

    s.merge();
    s.push_merged();

    BOOST_CHECK_NO_THROW(m = c0.fetch_merged("m"));

    BOOST_CHECK_SMALL(cuv::norm1(m-6.f), 0.001f);
}

struct optimizer{
    typedef boost::shared_ptr<cuvnet::Op>     op_ptr;
    float lossval;

    void plain_gd(){
        boost::shared_ptr<cuvnet::ParameterInput> inp1(new cuvnet::ParameterInput(cuv::extents[5][6]));
        boost::shared_ptr<cuvnet::ParameterInput> inp2(new cuvnet::ParameterInput(cuv::extents[5][6]));
        op_ptr loss = cuvnet::mean(cuvnet::pow(inp1-inp2, 2.f));
        cuvnet::function f(loss,0);

        {   // plain GD
            cuv::fill_rnd_uniform(inp1->data());
            cuv::fill_rnd_uniform(inp2->data());
            std::vector<cuvnet::Op*> params;
            params.push_back(inp1.get());
            std::cout << "plain GD: before optimization: " << f.evaluate()[0] << std::endl;
            cuvnet::gradient_descent gd(loss,0,params, 0.1f);

            gd.batch_learning(500, INT_MAX);
            std::cout << "plain GD: after optimization: " << f.evaluate()[0] << std::endl;
            BOOST_CHECK_SMALL(lossval = f.evaluate()[0], 0.001f);
        }
    }

    void async_gd(unsigned int i){
        boost::shared_ptr<cuvnet::ParameterInput> inp1(new cuvnet::ParameterInput(cuv::extents[5][6],"inp1"));
        boost::shared_ptr<cuvnet::ParameterInput> inp2(new cuvnet::ParameterInput(cuv::extents[5][6],"inp2"));
        op_ptr loss = cuvnet::mean(cuvnet::pow(inp1-inp2, 2.f));
        cuvnet::function f(loss,0);

        {   
            cuv::fill_rnd_uniform(inp1->data());
            //cuv::fill_rnd_uniform(inp2->data()); // 'teacher' should be same for all instances
            inp2->data() = 0.85f;
            std::vector<cuvnet::Op*> params;
            params.push_back(inp1.get());
            std::cout << "async GD: before optimization: " << f.evaluate()[0] << std::endl;
            cuvnet::diff_recording_gradient_descent<cuvnet::gradient_descent> gd(loss,0,params, 0.1f);

            using namespace cuvnet::network_communication;
            std::string cid = "client-"+boost::lexical_cast<std::string>(i);
            client c(HOST,DB,KEY,cid);
            param_synchronizer ps(cid, c,5,1,0,0,params);
            gd.set_sync_function(boost::ref(ps));

            gd.batch_learning(500, INT_MAX);
            std::cout << "async GD: after optimization: " << (lossval = f.evaluate()[0]) << std::endl;
            BOOST_CHECK_SMALL((float)f.evaluate()[0], 0.001f);
        }
    }

};

BOOST_AUTO_TEST_CASE( nc_gd ){
    // plain GD
    optimizer opt;
    opt.plain_gd();

    // async GD
    static const int n_clt = 2;
    optimizer*      clients[n_clt];
    boost::thread*  threads[n_clt];
    cuvnet::network_communication::server s(HOST,DB,KEY);
    s.cleanup();

    for (int i = 0; i < n_clt; ++i)
    {
        clients[i] = new optimizer();
        threads[i] = new boost::thread(boost::bind(&optimizer::async_gd, clients[i], i));
    }
    s.run(100,1000);
    for (int i = 0; i < n_clt; ++i) {
        threads[i]->join();
    }
    for (int i = 0; i < n_clt; ++i) {
        BOOST_CHECK_GT(opt.lossval, clients[i]->lossval);
    }
}

BOOST_AUTO_TEST_SUITE_END()
