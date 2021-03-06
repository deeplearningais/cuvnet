#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/network_communication.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/function.hpp>

#define V(X) #X<<":"<<(X)<<", "
#define HOST "131.220.7.92"
#define DB   "testnc"
#define KEY  "tst"


BOOST_AUTO_TEST_SUITE( netcom )
BOOST_AUTO_TEST_CASE( server_start ){
   using namespace cuvnet;
   using namespace network_communication;
   cuvnet::network_communication::merger mrg(1.f);
   server s(HOST,DB,KEY, &mrg);
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

    float learnrate = 1.f;
    cuvnet::network_communication::merger mrg(learnrate);
    client c(HOST,DB,KEY);
    server s(HOST,DB,KEY, &mrg);
    s.cleanup();

    BOOST_CHECK_THROW(c.fetch_merged("m"), value_not_found_exception);

    cuv::tensor<float, cuv::host_memory_space> n, m(cuv::extents[5][6]), md;

    cuv::sequence(m);
    md = m.copy();
    md *= learnrate;

    c.put_for_merging("m", md, m); // only puts m and md

    s.merge();
    s.push_merged();

    BOOST_CHECK_NO_THROW(n = c.fetch_merged("m"));

    BOOST_REQUIRE(n.ptr() != NULL);
    
    BOOST_CHECK_SMALL(cuv::norm1((m-md)-n), 0.001f);
}

BOOST_AUTO_TEST_CASE( averaging ){
    using namespace cuvnet;
    using namespace network_communication;

    client c0(HOST,DB,KEY, "c0");
    client c1(HOST,DB,KEY, "c1");
    cuvnet::network_communication::merger mrg(1.f);
    server s(HOST,DB,KEY, &mrg);
    s.cleanup();

    BOOST_CHECK_THROW(c0.fetch_merged("m"), value_not_found_exception);
    BOOST_CHECK_THROW(c1.fetch_merged("m"), value_not_found_exception);

    cuv::tensor<float, cuv::host_memory_space> m(cuv::extents[5][6]), md;
    m = 3.f;
    md = m.copy();
    md = -2.f;

    c0.put_for_merging("m", md, m); // puts m and md

    md = -4.f;

    c1.put_for_merging("m", md, m); // puts md

    s.merge();
    s.push_merged();

    m = c0.fetch_merged("m");
    m -= 9.f;

    BOOST_CHECK_SMALL(cuv::norm1(m), 0.001f);
}

BOOST_AUTO_TEST_CASE( self_overwrite ){
    // a client writes twice before averaging: both will be merged!
    using namespace cuvnet;
    using namespace network_communication;

    client c0(HOST,DB,KEY);
    cuvnet::network_communication::merger mrg(1.f);
    server s(HOST,DB,KEY, &mrg);
    s.cleanup();

    BOOST_CHECK_THROW(c0.fetch_merged("m"), value_not_found_exception);

    cuv::tensor<float, cuv::host_memory_space> m(cuv::extents[5][6]), md;
    m = 2.f;
    md = m.copy();
    md = -3.f;

    c0.put_for_merging("m", md, m); // puts m and md

    md = -4.f;

    c0.put_for_merging("m", md, m); // puts=adds new md

    s.merge();
    s.merge();
    s.push_merged();

    BOOST_CHECK_NO_THROW(m = c0.fetch_merged("m"));

    BOOST_CHECK_SMALL(cuv::norm1(m-9.f), 0.001f);
}

void sleeper(){ 
    boost::this_thread::sleep(boost::posix_time::milliseconds(10 + 10 * drand48()));
}
struct optimizer{
    typedef boost::shared_ptr<cuvnet::Op>     op_ptr;
    float lossval;


    void plain_gd(){
        using cuvnet::Noiser;
        boost::shared_ptr<cuvnet::ParameterInput> inp1(new cuvnet::ParameterInput(cuv::extents[5][6]));
        boost::shared_ptr<cuvnet::ParameterInput> inp2(new cuvnet::ParameterInput(cuv::extents[5][6]));
        //cuv::fill_rnd_uniform(inp1->data());
        inp1->data() = 0.0f;

        // input 1 is just noise, this is our parameter vector.
        // we will try to get to some target value stored in inp2:
        inp2->data() = 0.5f;

        // but we cannot see inp2 directly, we only see its noisy version

        boost::shared_ptr<Noiser> noisy_inp2 = boost::dynamic_pointer_cast<Noiser>(cuvnet::add_rnd_normal(inp2, 0.2f));
        op_ptr loss = cuvnet::mean(cuvnet::pow(noisy_inp2-inp1, 2.f));
        cuvnet::function f(loss,0);

        {   
            std::vector<cuvnet::Op*> params;
            params.push_back(inp1.get());
            noisy_inp2->set_active(false);
            std::cout << "plain GD: before optimization: " << f.evaluate()[0] << std::endl;
            noisy_inp2->set_active(true);
            cuvnet::gradient_descent gd(loss,0,params, 0.05f);

            gd.batch_learning(100, INT_MAX);
            noisy_inp2->set_active(false);
            std::cout << "plain GD: after optimization: " << (lossval = f.evaluate()[0]) << std::endl;
            BOOST_CHECK_SMALL((float)f.evaluate()[0], 0.05f);
        }
    }

    void async_gd(unsigned int i){
        cuv::initCUDA(1);
        cuv::initialize_mersenne_twister_seeds(i);
        using cuvnet::Noiser;
        boost::shared_ptr<cuvnet::ParameterInput> inp1(new cuvnet::ParameterInput(cuv::extents[5][6],"inp1"));
        boost::shared_ptr<cuvnet::ParameterInput> inp2(new cuvnet::ParameterInput(cuv::extents[5][6],"inp2"));
        //cuv::fill_rnd_uniform(inp1->data());
        inp1->data() = 0.0f;

        // input 1 is just noise, this is our parameter vector.
        // we will try to get to some target value stored in inp2:
        inp2->data() = 0.5f;

        // but we cannot see inp2 directly, we only see its noisy version

        boost::shared_ptr<Noiser> noisy_inp2 = boost::dynamic_pointer_cast<Noiser>(cuvnet::add_rnd_normal(inp2, 0.2f));
        op_ptr loss = cuvnet::mean(cuvnet::pow(noisy_inp2-inp1, 2.f));
        cuvnet::function f(loss,0);

        {   
            std::vector<cuvnet::Op*> params;
            params.push_back(inp1.get());
            noisy_inp2->set_active(false);
            std::cout << "async GD: before optimization: " << f.evaluate()[0] << std::endl;
            noisy_inp2->set_active(true);
            cuvnet::diff_recording_gradient_descent<cuvnet::gradient_descent> gd(loss,0,params, 0.05f);

            using namespace cuvnet::network_communication;
            std::string cid = "client-"+boost::lexical_cast<std::string>(i);
            client c(HOST,DB,KEY,cid);
            param_synchronizer ps("stage", c, 1, 1, 0, 0, params);
            gd.set_sync_function(boost::ref(ps));
            gd.after_epoch.connect(boost::bind(sleeper)); // artificially slow down to allow server to catch up

            gd.batch_learning(100, INT_MAX);
            noisy_inp2->set_active(false);
            std::cout << "async GD: after optimization: " << (lossval = f.evaluate()[0]) << std::endl;
            BOOST_CHECK_SMALL((float)f.evaluate()[0], 0.05f);
        }
    }

};

/*
BOOST_AUTO_TEST_CASE( nc_gd ){
    // plain GD
    optimizer opt;
    opt.plain_gd();

    // async GD
    static const int n_clt = 3;
    optimizer*      clients[n_clt];
    boost::thread*  threads[n_clt];
    using cuvnet::network_communication::server;
    //cuvnet::network_communication::adagrad_merger mrg(.05f, 0.1f);
    cuvnet::network_communication::momentum_merger mrg(1.00, 0.1f);
    //cuvnet::network_communication::merger mrg(0.05f);

    server s(HOST,DB,KEY, &mrg);
    s.cleanup();

    for (int i = 0; i < n_clt; ++i)
    {
        std::cout << "starting client #" << i << std::endl;
        clients[i] = new optimizer();
        threads[i] = new boost::thread(boost::bind(&optimizer::async_gd, clients[i], i));
    }
    boost::thread server_thread(boost::bind(&server::run,&s,10,-1));

    for (int i = 0; i < n_clt; ++i) {
        threads[i]->join();
    }
    s.request_stop();
    server_thread.join();
    for (int i = 0; i < n_clt; ++i) {
        BOOST_CHECK_GT(opt.lossval + 0.001, clients[i]->lossval);
    }
}
*/

BOOST_AUTO_TEST_SUITE_END()
