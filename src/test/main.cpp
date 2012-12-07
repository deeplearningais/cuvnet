#include <iostream>
#include <fstream>
#include <cuv.hpp>
#include <cuvnet/common.hpp>

#define BOOST_TEST_MODULE t_cuvnet
//#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include <cuvnet/tools/logging.hpp>


std::ofstream os;

struct FooEnvironment 
{
    FooEnvironment() {
        cuvnet::Logger log("test_log.xml");
        os.open("test_results.txt");
        boost::unit_test::results_reporter::set_stream(os);
        cuv::initCUDA(0);
        if(cuv::IsSame<cuv::dev_memory_space, cuvnet::matrix::memory_space_type>::Result::value){
            cuv::initialize_mersenne_twister_seeds();
        }
    }
  ~FooEnvironment(){}
};

BOOST_GLOBAL_FIXTURE(FooEnvironment);

//int main(int argc, char **argv) {
    //return ::boost::unit_test::unit_test_main( &init_function, argc, argv );
//}

