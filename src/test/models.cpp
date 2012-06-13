#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>

#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/denoising_auto_encoder.hpp>
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/generic_regression.hpp>


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
       simple_auto_encoder<simple_auto_encoder_weight_decay> ae(4, true, 0.01f);
       ae.init(inp);
       derivative_tester(*ae.loss(),0,false,.01f, 0.f, 1.f); // generate inputs in interval 0,1
   }

   {
       simple_auto_encoder<simple_auto_encoder_weight_decay> ae(4, false, 0.01f);
       ae.init(inp);
       derivative_tester(*ae.loss(),0,false,.01f);
   }
}

TEST_F(RandomNumberUsingTest, denoising_ae_loss_derivative){
   boost::shared_ptr<Op>  inp = boost::make_shared<Input>(cuv::extents[3][5]);

   denoising_auto_encoder<simple_auto_encoder_weight_decay> ae(4, true, 0.0f, 0.01f); // zero noise
   ae.init(inp);
   derivative_tester(*ae.loss());
}

TEST_F(RandomNumberUsingTest, stack_derivative){
   boost::shared_ptr<Op>  inp = boost::make_shared<Input>(cuv::extents[3][5]);

   auto_encoder_stack<> ae(true); // zero noise

   typedef denoising_auto_encoder<simple_auto_encoder_weight_decay> ae_type;
   ae.add<ae_type>(4, true, 0.0f, 0.01f); 
   ae.add<ae_type>(4, true, 0.0f, 0.01f);

   ae.init(inp);

   derivative_tester(*ae.loss());
}


TEST_F(RandomNumberUsingTest, logistic_reg_loss_derivative){
   boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
   boost::shared_ptr<Input>  target = boost::make_shared<Input>(cuv::extents[3][5]);
   target->set_derivable(false);

   logistic_regression lg(inp, target); 
   derivative_tester(*lg.loss(), 0, false, .01);
}

TEST_F(RandomNumberUsingTest, linear_reg_loss_derivative){
   boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
   boost::shared_ptr<Input>  target = boost::make_shared<Input>(cuv::extents[3][5]);
   target->set_derivable(false);

   linear_regression lg(inp, target); 
   derivative_tester(*lg.loss(), 0, false, .01);
}

