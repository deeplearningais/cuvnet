#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>

#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/denoising_auto_encoder.hpp>
using namespace cuvnet::derivative_testing;

class RandomNumberUsingTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
      if(cuv::IsSame<cuv::dev_memory_space, matrix::memory_space_type>::Result::value){
          cuv::initCUDA(0);
          cuv::initialize_mersenne_twister_seeds();
      }
  }
};


TEST_F(RandomNumberUsingTest, simple_ae_loss_derivative){
   boost::shared_ptr<Op>  inp = boost::make_shared<Input>(cuv::extents[3][5]);

   {
       simple_auto_encoder ae(inp, 4, true);
       derivative_tester(*ae.loss(),0,false,.01f, 0.f, 1.f); // generate inputs in interval 0,1
   }

   {
       simple_auto_encoder ae(inp, 4, false);
       derivative_tester(*ae.loss(),0,false,.01f);
   }
}

TEST_F(RandomNumberUsingTest, denoising_ae_loss_derivative){
   boost::shared_ptr<Op>  inp = boost::make_shared<Input>(cuv::extents[3][5]);

   denoising_auto_encoder ae(inp, 4, true, 0.0f); // zero noise
   derivative_tester(*ae.loss());
}
