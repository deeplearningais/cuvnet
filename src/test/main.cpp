//#include <glog/logging.h>
#include <cuv.hpp>
#include <cuvnet/common.hpp>

#define BOOST_TEST_MODULE t_cuvnet
//#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>



struct FooEnvironment 
{
  FooEnvironment() {
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

