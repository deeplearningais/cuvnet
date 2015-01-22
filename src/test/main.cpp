#include <cstdlib>     /* getenv */
#include <iostream>
#include <fstream>
#include <cuv.hpp>
#include <cuvnet/common.hpp>
#include <cuvnet/smart_ptr.hpp>

#define BOOST_TEST_MODULE t_cuvnet
//#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
#include <cuvnet/tools/logging.hpp>


#ifndef NO_THEANO_WRAPPERS
#      include <cuv/convolution_ops/convolution_ops_theano.hpp>
#endif

std::ofstream os;

struct FooEnvironment
{
    FooEnvironment() {
        cuvnet::Logger log("test_log.xml");
        os.open("test_results.txt");
        boost::unit_test::results_reporter::set_stream(os);
        int dev = -1;
        char* device = std::getenv("DEV");
        if(device != NULL)
            dev = boost::lexical_cast<int>(std::string(device));
        std::cerr << "Initializing CUDA device " << dev << std::endl;
        cuv::initCUDA(dev);
        if(cuv::IsSame<cuv::dev_memory_space, cuvnet::matrix::memory_space_type>::Result::value){
            cuv::initialize_mersenne_twister_seeds();
        }
#ifndef NO_THEANO_WRAPPERS
        cuv::theano_conv::initcuda();
#endif
    }
  ~FooEnvironment(){
      cuv::safeThreadSync();
      //cuvnet::cow_ptr<cuvnet::matrix>::s_allocator.reset();
      cuvnet::cow_ptr<cuvnet::matrix>::s_allocator.reset(
              new cuv::pooled_cuda_allocator());
          //new cuv::default_allocator());
#ifndef NO_THEANO_WRAPPERS
     cuv::theano_conv::finalize_cuda();
#endif

  }
};

BOOST_GLOBAL_FIXTURE(FooEnvironment);

//int main(int argc, char **argv) {
    //return ::boost::unit_test::unit_test_main( &init_function, argc, argv );
//}

