//#include <glog/logging.h>
#include <cuv.hpp>
#include <cuvnet/common.hpp>
#include <gtest/gtest.h>

class FooEnvironment 
: public ::testing::Environment
{
 public:
  virtual void SetUp() {
     cuv::initCUDA(0);
      if(cuv::IsSame<cuv::dev_memory_space, cuvnet::matrix::memory_space_type>::Result::value){
          cuv::initialize_mersenne_twister_seeds();
      }
  }
  virtual void TearDown() {}
};

int main(int argc, char **argv) {
  //google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::Environment* const foo_env 
      = ::testing::AddGlobalTestEnvironment(new FooEnvironment);
  return RUN_ALL_TESTS();
}

