#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>

#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/denoising_auto_encoder.hpp>
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/convolutional_auto_encoder.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/object_detector.hpp>
#include <cuvnet/models/lenet.hpp>
#include <tools/monitor.hpp>
#include <tools/function.hpp>
#include <cuvnet/models/relational_auto_encoder.hpp>


using namespace cuvnet::derivative_testing;

TEST(Function, simple){
    boost::shared_ptr<ParameterInput> inp(new ParameterInput(cuv::extents[3][5]));
    boost::shared_ptr<Op> func(new Sum(inp->result()));
    boost::shared_ptr<Sink> out(new Sink("out",func->result()));

    cuvnet::function func2(out + 1, 0, "func2");

    inp->data() = 1.f;
    swiper s(*func,0,std::vector<Op*>());

    s.fprop();
    EXPECT_EQ(15, out->cdata()[0]);

    // change the input values, this should not affect the expected result
    // since the value in the sink is reused
    inp->data() = 2.f; 

    func2.evaluate();
    EXPECT_EQ(16, func2.result()[0]);

    // now do a second sweep on the inputs, which should yield 30 now
    s.fprop();
    EXPECT_EQ(30, out->cdata()[0]);

    // ...and recalculate the function, which should give us 31.
    func2.evaluate();
    EXPECT_EQ(31, func2.result()[0]);

}
TEST(Monitor, simple){
    boost::shared_ptr<ParameterInput> inp(new ParameterInput(cuv::extents[3][5]));
    boost::shared_ptr<Op> func(new Sum(inp->result()));
    inp->data() = 1.f;

    {
        cuvnet::monitor mon;
        EXPECT_EQ(0,func->result(0)->result_uses.size());
        mon.add(monitor::WP_SCALAR_EPOCH_STATS, func, "sum");
        EXPECT_EQ(1,func->result(0)->result_uses.size());
    
        swiper swp(*func,0,std::vector<Op*>());
        swp.fprop();
    
        EXPECT_EQ(15, mon["sum"][0]);
    }

    // test destruction of sinks when monitor is destroyed
    EXPECT_EQ(0,func->result(0)->result_uses.size());
}

TEST(Monitor, function){
    boost::shared_ptr<ParameterInput> inp(new ParameterInput(cuv::extents[3][5]));
    boost::shared_ptr<Op> func(new Sum(inp->result()));
    boost::shared_ptr<Sink> out(new Sink("out",func->result()));

    cuvnet::monitor mon;
    mon.add(monitor::WP_FUNC_SINK, out+1, "func2");

    inp->data() = 1.f;
    swiper s(*func,0,std::vector<Op*>());

    s.fprop();
    EXPECT_EQ(15, out->cdata()[0]);

    // change the input values, this should not affect the expected result
    // since the value in the sink is reused
    inp->data() = 2.f; 

    EXPECT_EQ(16, mon["func2"][0]);

    // now do a second sweep on the inputs, which should yield 30 now
    s.fprop();
    EXPECT_EQ(30, out->cdata()[0]);

    // ...and recalculate the function, which should give us 31.
    EXPECT_EQ(31, mon["func2"][0]);

}

class RandomNumberUsingTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
      // mersenne twister now initialized in Environment::SetUp, see main.cpp
  }
};


TEST_F(RandomNumberUsingTest, simple_ae_loss_derivative){
   boost::shared_ptr<Op>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);

   {
       simple_auto_encoder<simple_auto_encoder_weight_decay> ae(4, true);
       ae.init(inp, .01f);
       derivative_tester(*ae.loss(),0,false,.01f, 0.f, 1.f); // generate inputs in interval 0,1
   }

   {
       simple_auto_encoder<simple_auto_encoder_weight_decay> ae(4, false);
       ae.init(inp, .01f);
       derivative_tester(*ae.loss(),0,false,.01f);
   }
}
TEST_F(RandomNumberUsingTest, convolutional_auto_encoder_derivative ){
    // nBatch x nChannels x nPixels
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[2][2][6*6], "input");
   cuv::fill_rnd_uniform(inp->data());
   inp->data() -= 0.5f;
   //inp->set_derivable(false);

   {
       conv_auto_encoder<simple_auto_encoder_no_regularization> ae(false, 3, 2);
       ae.init(inp);

       std::cout << "-encoded" << std::endl;
       derivative_tester(*ae.get_encoded(),0,true,.01f, 0,0); // generate inputs in interval 0,1

       std::cout << "-decoded" << std::endl;
       derivative_tester(*ae.get_decoded(),0,true,.1f, 0,0); // generate inputs in interval 0,1

       std::cout << "-loss" << std::endl;
       derivative_tester(*ae.loss(),0,true,.1f, 0,0); // generate inputs in interval 0,1
   }
}

TEST_F(RandomNumberUsingTest, denoising_ae_loss_derivative){
   boost::shared_ptr<Op>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);

   denoising_auto_encoder<simple_auto_encoder_weight_decay> ae(4, true, .0f); // zero noise
   ae.init(inp, 0.00f);
   derivative_tester(*ae.loss(),0,true,.01);
}

TEST_F(RandomNumberUsingTest, stack_derivative){
   boost::shared_ptr<Op>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);

   auto_encoder_stack<> ae(true); // zero noise

   typedef denoising_auto_encoder<simple_auto_encoder_weight_decay> ae_type;
   ae.add<ae_type>(4, true, .0f); 
   ae.add<ae_type>(4, true, .0f);

   ae.init(inp, 0.01f);

   derivative_tester(*ae.loss(), 0, true, .01);
}

TEST_F(RandomNumberUsingTest, linear_regression_derivative){
   boost::shared_ptr<ParameterInput>  inp    = boost::make_shared<ParameterInput>(cuv::extents[3][5], "input");
   boost::shared_ptr<ParameterInput>  target = boost::make_shared<ParameterInput>(cuv::extents[3][5], "target");

   linear_regression lg(inp, target); 
   derivative_tester(*lg.get_loss(), 0, false, .01);
}

TEST_F(RandomNumberUsingTest, logistic_regression_derivative){
   boost::shared_ptr<ParameterInput>  inp    = boost::make_shared<ParameterInput>(cuv::extents[3][5], "input");
   boost::shared_ptr<ParameterInput>  target = boost::make_shared<ParameterInput>(cuv::extents[3][5], "target");
   target->set_derivable(false); // cannot derive for target of logistic regression

   logistic_regression lg(inp, target); 
   derivative_tester(*lg.get_loss(), 0, false, .03, 0.1, 0.9);
}

TEST_F(RandomNumberUsingTest, obj_det){
   boost::shared_ptr<ParameterInput>  inp    = boost::make_shared<ParameterInput>(cuv::extents[2][1][28*28], "input");
   boost::shared_ptr<ParameterInput>  ign    = boost::make_shared<ParameterInput>(cuv::extents[2][16][28*28], "ign");
   boost::shared_ptr<ParameterInput>  target = boost::make_shared<ParameterInput>(cuv::extents[2][16][28*28], "target");
   target->set_derivable(false); // cannot derive for target of logistic regression

   cuv::fill_rnd_uniform(inp->data());
   cuv::fill_rnd_uniform(target->data());
   cuv::fill_rnd_uniform(ign->data());
   inp->data() -= 0.5f;

   obj_detector od(5,16,5,16); 
   od.init(inp,ign,target);
   std::cout << "---hl2--" << std::endl;
   derivative_tester(*od.hl2, 0, true, .1, 0, 0);

   std::cout << "---hl1--" << std::endl;
   derivative_tester(*od.hl1, 0, true, .1, 0, 0);

   std::cout << "---loss--" << std::endl;
   derivative_tester(*od.get_loss(), 0, true, .1, 0, 0);
}

TEST_F(RandomNumberUsingTest, lenet_derivative){
   boost::shared_ptr<ParameterInput>  inp    = boost::make_shared<ParameterInput>(cuv::extents[2][1][28*28], "input");
   boost::shared_ptr<ParameterInput>  target = boost::make_shared<ParameterInput>(cuv::extents[2][10], "target");
   cuv::fill_rnd_uniform(inp->data());
   cuv::fill_rnd_uniform(target->data());
   inp->data() -= 0.5f;

   target->set_derivable(false); // cannot derive for target of logistic regression

   lenet ln(5,16,5,16,2); 
   ln.init(inp,target);
   std::cout << "---hl2--" << std::endl;
   derivative_tester(*ln.hl2, 0, false, .15, 0,0);

   std::cout << "---hl3--" << std::endl;
   derivative_tester(*ln.hl3, 0, false, .1, 0,0);

   std::cout << "---hl1--" << std::endl;
   derivative_tester(*ln.hl1, 0, false, .1, 0,0);

   std::cout << "---loss--" << std::endl;
   derivative_tester(*ln.get_loss(), 0, false, .1, 0,0);
}

TEST_F(RandomNumberUsingTest, relational_auto_encoder_derivative){
   boost::shared_ptr<Op>  inp_x = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   boost::shared_ptr<Op>  inp_y = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   relational_auto_encoder r_ae(4, 4,  false); 
   r_ae.init(inp_x, inp_y);

   derivative_tester(*r_ae.loss(), 0, true, .01);
}
