#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <boost/assign.hpp>

#define CUVNET_PRECISE_SUM 1

#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <cuvnet/tools/function.hpp>

#include <cuvnet/ops.hpp>

#include <boost/test/unit_test.hpp>
#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#define EXPECT_NEAR(X,Y,D) BOOST_REQUIRE_LT(((X)-(Y))*((X)-(Y)), ((D)*(D)))
#endif

using namespace cuvnet;
using std::printf;
using namespace cuvnet::derivative_testing;

BOOST_AUTO_TEST_SUITE( op_test )

BOOST_AUTO_TEST_CASE(deltasink){
    typedef boost::shared_ptr<Op> ptr_t;
    typedef boost::shared_ptr<ParameterInput> param_t;
    param_t inp = boost::make_shared<ParameterInput>(cuv::extents[2][4]);
    fill_rnd_uniform(inp->data());
    ptr_t func  = boost::make_shared<Pow>(2.f,inp->result());
    boost::shared_ptr<DeltaSink> ds = delta_sink("pow_delta", func); // monitor the delta

    swiper s(*func, 0, boost::assign::list_of<Op*>(inp.get()));
    s.fprop();
    s.bprop();
    for (unsigned int i = 0; i < 2*4; ++i) {
        EXPECT_NEAR(ds->cdata()[i], 2 * inp->data()[i], 0.001f);
    }
    s.fprop();
    s.bprop();
    for (unsigned int i = 0; i < 2*4; ++i) {
        EXPECT_NEAR(ds->cdata()[i], 2 * inp->data()[i], 0.001f);
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( derivative_test )
BOOST_AUTO_TEST_CASE(derivative_test_pipe){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Pipe>(inp->result(),0);
   derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_rowrange_select){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);
   boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);

   ptr_t func                    = row_range_select(inp0,inp1,2,0); // select 2 of the rows (fix to 0-2 for testing)

   derivative_tester(*result(func,0));
   derivative_tester(*result(func,1));
}
BOOST_AUTO_TEST_CASE(derivative_test_row_select){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);

   ptr_t func                    = row_select(inp0,inp1,1);

   derivative_tester(*result(func,0));
   derivative_tester(*result(func,1));
}
BOOST_AUTO_TEST_CASE(derivative_test_scalar_like){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                     = boost::make_shared<ScalarLike>(inp->result(), 3.4f);
   derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_pow){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());

   function f(pow, 0);
   matrix m = f.evaluate();
   for(unsigned int i = 0; i < inp->data().size(); i++){
       EXPECT_NEAR(m[i], inp->data()[i] * inp->data()[i], 0.01);
   }

   derivative_tester(*pow);
}
BOOST_AUTO_TEST_CASE(derivative_test_exp){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t exp                     = boost::make_shared<Exp>(2,inp->result());
   derivative_tester(*exp,0,false,0.01);
}
BOOST_AUTO_TEST_CASE(derivative_test_abs){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Abs>(inp->result());
   derivative_tester(*func);
}

BOOST_AUTO_TEST_CASE(derivative_test_log){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   boost::shared_ptr<Pow>    pw  = boost::make_shared<Pow>(2, inp->result());
   ptr_t func                    = boost::make_shared<Log>(pw->result());
   derivative_tester(*func);
}

BOOST_AUTO_TEST_CASE(derivative_test_mean){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Mean>(inp->result());
    derivative_tester(*func);
}

BOOST_AUTO_TEST_CASE(derivative_test_tanh){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Tanh>(inp->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_sin){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Sin>(inp->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_cos){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Cos>(inp->result());
    derivative_tester(*func);
}

BOOST_AUTO_TEST_CASE(derivative_test_add_scalar){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<AddScalar>(1.f,inp->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_mult_scalar){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<MultScalar>(1.5f,inp->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_subtract_from_scalar){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    //ptr_t func                    = boost::make_shared<SubtractFromScalar>(1.f,inp->result());
    ptr_t func                    = 1.f-inp;
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_rectified_linear){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<RectifiedLinear>(inp->result());
    derivative_tester(*func);
    derivative_tester(*func, 1);
}

BOOST_AUTO_TEST_CASE(derivative_test_multiply){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Multiply>(inp0->result(), inp1->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_atan2){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5], "y");
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5], "x");
    ptr_t func                     = boost::make_shared<Atan2>(inp0->result(), inp1->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_eps_insensitive_loss){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    inp0->set_derivable(false);
    ptr_t func                     = boost::make_shared<EpsilonInsensitiveLoss>(0.1, inp0->result(), inp1->result());
    {
        inp0->data()[0] = 0.2f;
        inp1->data()[0] = 0.3f;
        function f(func, 0);
        EXPECT_NEAR(f.evaluate()[0], 0, 0.01);
    }
    derivative_tester(*func, 0, true, .003, 0.0, 1.0);
}
BOOST_AUTO_TEST_CASE(derivative_test_hinge_loss){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    inp0->set_derivable(false);
    ptr_t func                     = boost::make_shared<HingeLoss>(inp0->result(), inp1->result(), false);
    derivative_tester(*func, 0, true, .003, -1.0, 1.0);
}
BOOST_AUTO_TEST_CASE(derivative_test_squared_hinge_loss){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    inp0->set_derivable(false);
    ptr_t func                     = boost::make_shared<HingeLoss>(inp0->result(), inp1->result(), true);
    derivative_tester(*func, 0, true, .003, -1.0, 1.0);
}
BOOST_AUTO_TEST_CASE(derivative_test_neg_cross_entropy_logistic){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5]);
    ptr_t func                     = boost::make_shared<NegCrossEntropyOfLogistic>(inp0->result(), inp1->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(bernoulli_kl){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5], "inp0");
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5], "inp1");
        inp0->set_derivable(false);
        ptr_t func                     = boost::make_shared<BernoulliKullbackLeibler>(inp0->result(), inp1->result());
        derivative_tester(*func,0,false, .003, 0.1, 0.9);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5], "inp0");
        ptr_t func                     = boost::make_shared<BernoulliKullbackLeibler>(0.5f, inp0->result());
        derivative_tester(*func,0,false,.003,0.1,0.9); // test in the range of 0.1, 0.9
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_axpby){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.3, -2.5);
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_axpby_broadcast){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5], "inp0");
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[1], "scalar");
    ptr_t func                  = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.2, -2.6);
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_sum_mat_to_vec_squared){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0,false,true);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,false,true);
        derivative_tester(*func);
    }
    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
       ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2,false,true);
       derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,false,true);
      derivative_tester(*func);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_sum_mat_to_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0);
      derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1);
      derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][4][5]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1, true, true);
      derivative_tester(*func);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mean_mat_to_vec_squared){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0,true,true);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true,true);
        derivative_tester(*func);
    }
    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
       ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2,true,true);
       derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true,true);
      derivative_tester(*func);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mean_mat_to_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0,true);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true);
        derivative_tester(*func);
    }
    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
       ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2,true);
       derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true);
      derivative_tester(*func);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_sum_mat_to_vec3d){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        //std::cout << "axis=0" << std::endl;
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0);
        derivative_tester(*func);
    }
    {
        //std::cout << "axis=1" << std::endl;
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2);
        derivative_tester(*func);
    }
}
/*
 * // does not make much sense to test this
 *BOOST_AUTO_TEST_CASE(derivative_test_noiser){
 *        cuv::initialize_mersenne_twister_seeds();
 *        typedef boost::shared_ptr<Op> ptr_t;
 *    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);
 *    ptr_t func                     = boost::make_shared<Noiser>(inp0->result());
 *    derivative_tester(*func);
 *}
 */
BOOST_AUTO_TEST_CASE(derivative_test_sum){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[2][2]);
    ptr_t func                     = boost::make_shared<Sum>(inp0->result());
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_transpose){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[2][3]);
    ptr_t func                     = boost::make_shared<Transpose>(inp0->result());
    derivative_tester(*func);
}

BOOST_AUTO_TEST_CASE(derivative_test_prod){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5][3]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][8]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result());
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][8]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t');
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5][3]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'n','t');
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
        ptr_t func0		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
        ptr_t func1		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
        ptr_t func 		       = boost::make_shared<Axpby>(func0->result(), func1->result(), 1.3,1.5);
        derivative_tester(*func,0,false,0.03);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_sq_axpby){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp0->result(), 1.5f, -1.8f);
    func                           = boost::make_shared<Pow>(2.f,func->result());
    derivative_tester(*func);
}

BOOST_AUTO_TEST_CASE(derivative_test_xtx){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = inp0*inp0;
    derivative_tester(*func);
}
BOOST_AUTO_TEST_CASE(derivative_test_xt1mx){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = inp0*(1.f-inp0);
    derivative_tester(*func);
}

BOOST_AUTO_TEST_CASE(derivative_test_add_to_param){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t tmp0                     = boost::make_shared<Pow>(2.f, inp0->result());
    ptr_t tmp1                     = boost::make_shared<Pow>(3.f, inp0->result());
    ptr_t func                     = boost::make_shared<Sum>(tmp0->result());
    add_to_param(func, tmp1);  // sum(  x^2+x^3 )
    derivative_tester(*func, 0, false, 0.03f);
}

BOOST_AUTO_TEST_CASE(derivative_test_softmax){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 0);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 1);
        derivative_tester(*func);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_mll){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 1);
        derivative_tester(*func);
    }
    // higher dimensional input
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 3);
        derivative_tester(*func);
    }

    ///// SoftMax result of MultinomialLogisticLoss
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func,1);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 1);
        derivative_tester(*func,1);
    }
    // higher dimensional input
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func,1);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 3);
        derivative_tester(*func,1);
    }

}

BOOST_AUTO_TEST_CASE(derivative_test_mat_plus_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3]);
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 0);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][6]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][6]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [2]);
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 2);

        derivative_tester(*func);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mat_times_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 0);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [2]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 2);

        derivative_tester(*func);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mat_div_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    float prec = 0.0065;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 0);

        derivative_tester(*func, 0, false, prec);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func, 0, false, prec);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func, 0, false, prec);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [2]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 2);

        derivative_tester(*func, 0, false, prec);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_convolve){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;

    for (int padding = 0; padding < 2; ++padding)
    {
        {
            unsigned int nImgChan = 3;      // must be divisible by nGroups
            unsigned int nImgPixX = 16;
            unsigned int nImgPixY = 16;
            unsigned int nImg     = 4;
            unsigned int nGroups  = 1;      // must be divisible by 2 ??

            unsigned int nFiltChan = nImgChan/nGroups;
            unsigned int nFiltPixX  = 3;
            unsigned int nFilt     = 16;

            //unsigned int nResPix   = nImgPixX-nFiltPixX+1;

            {
                // sparse convolution
                unsigned int nGroups = 2;
                unsigned int nImgChan = 32;
                unsigned int nFiltChan = 16;
                unsigned int nFilt     = 32;
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt], "weights");
                boost::shared_ptr<Convolve> func        = boost::make_shared<Convolve>(inp0->result(), inp1->result(), 
                        padding, padding, 1, nGroups, 0);
                func->set_random_sparse();

                derivative_tester(*func,0,false,.03f);
            }

            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt], "weights");
                ptr_t func                       = boost::make_shared<Convolve>(inp0->result(), inp1->result(), 
                        padding, padding, 1, nGroups, 1);

                // it might not be possible to derive for images if they have only 3 channels!
                inp0->set_derivable(false);

                derivative_tester(*func,0,false,.03f);
            }

            {
                unsigned int nImgChan = 16;      // must be divisible by nGroups
                unsigned int nFiltChan = nImgChan/nGroups;
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt], "weights");
                ptr_t func                       = boost::make_shared<Convolve>(inp0->result(), inp1->result(), 
                        padding, padding, 1, nGroups, 1);

                derivative_tester(*func,0,false,.03f);
            }
        }

        {
            // reconstruction of auto-encoder... go from many "images" to one "filter".
            // this does not work in a straight-forward way, since alex' convs only
            // support n*16 outputs.
            // the version used here will use (temporarily) more memory and will be slower
            // (than a hypothetical "optimal" version)
            unsigned int nImgChan = 1;      // must be divisible by nGroups
            unsigned int nImgPixY  = 16;
            unsigned int nImgPixX  = 16;
            unsigned int nImg     = 16;
            unsigned int nGroups  = 1;      // must be divisible by 2 ??

            unsigned int nFiltChan = nImgChan/nGroups;
            unsigned int nFiltPixX  = 3;
            unsigned int nFilt     = 1;

            //unsigned int nResPix   = nImgPixX-nFiltPixX+1;
            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg],"inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt],"weights");
                ptr_t func                     = boost::make_shared<Convolve>(inp0->result(), inp1->result(), padding, 0, 1, 1);

                derivative_tester(*func,0,false,.03);
            }
        }
    }
}




void fill_with_permuted_sequence(matrix& m){
    cuv::sequence(m);
    cuv::tensor<float, cuv::host_memory_space> t = m;
    std::random_shuffle(t.ptr(), t.ptr() + t.size());
    m = t;
}

void test_derivative_test_tuple_ops(cuv::alex_conv::tuplewise_op_functor to){
    typedef boost::shared_ptr<Op> ptr_t;

    bool reinit = to != cuv::alex_conv::TO_MAX;
    using namespace cuv::alex_conv;

    {
       std::cout << "in first case tuplewise op" << std::endl;/* cursor */
       unsigned int sub_size = 3;
       unsigned int nImgChan = 2 * sub_size;      // must be divisible by nGroups
       unsigned int nImgPixX = 8;
       unsigned int nImgPixY = 8;
       unsigned int nImg     = 4;

       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
       ptr_t func   = boost::make_shared<Tuplewise_op>(inp0->result(), 0, sub_size, to, 0.0001f);

       fill_with_permuted_sequence(inp0->data());
       if(reinit)
           derivative_tester(*func);
       else
           derivative_tester(*func, 0, false, 0.03, 0, 0);
    }



    {
        std::cout << "in 2nd case tuplewise op" << std::endl;/* cursor */
        unsigned int sub_size = 3;
        unsigned int nImgChan = 2 * sub_size;      // must be divisible by nGroups
        unsigned int nImgPixX = 8;
        unsigned int nImgPixY = 8;
        unsigned int nImg     = 4;

        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgPixY][nImgPixX][nImg][nImgChan], "inputs");
        ptr_t func   = boost::make_shared<Tuplewise_op>(inp0->result(), 3, sub_size, to, 0.0001f);

       if(reinit)
           derivative_tester(*func);
       else
           derivative_tester(*func, 0, false, 0.03, 0, 0);
    }

    {
        std::cout << "in 3rd case tuplewise op" << std::endl;/* cursor */
        unsigned int sub_size = 3;
        unsigned int nImgChan = 2 * sub_size;      // must be divisible by nGroups
        unsigned int nImg     = 8;

        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan], "inputs");
        ptr_t func   = boost::make_shared<Tuplewise_op>(inp0->result(), 1, sub_size, to, 0.0001f);

        fill_with_permuted_sequence(inp0->data());
       if(reinit)
           derivative_tester(*func);
       else
           derivative_tester(*func, 0, false, 0.03, 0, 0);
    }

}
BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_norm){
    test_derivative_test_tuple_ops(cuv::alex_conv::TO_NORM);
}

BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_max){
    test_derivative_test_tuple_ops(cuv::alex_conv::TO_MAX);
}

BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_mean){
    test_derivative_test_tuple_ops(cuv::alex_conv::TO_MEAN);
}
BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_subsample){
    test_derivative_test_tuple_ops(cuv::alex_conv::TO_SUBSAMPLE);
}
BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_add_squared){
    test_derivative_test_tuple_ops(cuv::alex_conv::TO_ADD_SQUARED);
}

BOOST_AUTO_TEST_CASE(derivative_test_bed_of_nails){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 1;      // must be divisible by nGroups
    unsigned int nImgPixY  = 16;
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
    {
        ptr_t func		               = boost::make_shared<BedOfNails>(inp0->result());
        derivative_tester(*func);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_sep_conv){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 1;      // must be divisible by nGroups
    unsigned int nImgPixY  = 16;
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;

    matrix kernel(2*2+1);
    kernel = 1.f;

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
    {
        ptr_t func		               = boost::make_shared<SeparableFilter>(inp0->result(), kernel);
        derivative_tester(*func);
    }
    kernel = 0.f;
    kernel[kernel.size()/2] = 1.f;
    {
        ptr_t func		               = boost::make_shared<SeparableFilter>(inp0->result(), kernel);
        derivative_tester(*func);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_sep_conv1d){
    typedef boost::shared_ptr<Op> ptr_t;

    unsigned int nImgChan = 5;
    unsigned int nImgPixX  = 8;
    unsigned int nImg     = 5;

    cuv::tensor<float,cuv::host_memory_space> kernel(3);
    kernel(0) = 0;
    kernel(1) = -1;
    kernel(2) = 1;

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixX][nImg]);
        for (unsigned int i = 0; i < 3; i++)
        {
            if(i == 1)
                // this is currently broken in cuv and will yield an assertion in cuvnet.
                continue;
            //std::cout << " testing derivative of sep_conv_1d for dim " << i << std::endl;[> cursor <]
            ptr_t func		               = boost::make_shared<SeparableFilter1d>(inp0->result(), kernel, i);
            derivative_tester(*func);
        }

    }
}

/*
 *BOOST_AUTO_TEST_CASE(DISABLED_derivative_test_resize_bilinear){
 *    typedef boost::shared_ptr<Op> ptr_t;
 *
 *    using namespace cuv::alex_conv;
 *    unsigned int nImgChan = 1;      // must be divisible by nGroups
 *    unsigned int nImgPixY  = 16;
 *    unsigned int nImgPixX  = 16;
 *    unsigned int nImg     = 1;
 *
 *    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
 *    {
 *        ptr_t func		               = boost::make_shared<ResizeBilinear>(inp0->result(), 2.f);
 *        derivative_tester(*func);
 *    }
 *}
 */

BOOST_AUTO_TEST_CASE(derivative_test_response_normalization_cross_maps){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 16;      // must be divisible by 16, it seems
    unsigned int nImgPixY  = 16;
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
    for(unsigned int i=0; i<nImgChan; i++){
        for (unsigned int j = 0; j < nImg; ++j)
        {
            for (unsigned int y = 0; y < nImgPixY; ++y)
            {
                for (unsigned int x = 0; x < nImgPixX; ++x)
                {
                    //inp0->data()(i,y,x,j) = ((x%2)==0) && ((y%2)==1);
                    inp0->data()(i,y,x,j) = 0.1f + 0.9 * drand48();
                }
            }
        }
    }
    {
        ptr_t func		               = boost::make_shared<ResponseNormalizationCrossMaps>(inp0->result(), 3, 0.0000125f, 0.75f, false);
        derivative_tester(*func,0,true);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_response_normalization){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 16;      // must be divisible by 16, it seems
    unsigned int nImgPixY  = 16;
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
    for(unsigned int i=0; i<nImgChan; i++){
        for (unsigned int j = 0; j < nImg; ++j)
        {
            for (unsigned int y = 0; y < nImgPixY; ++y)
            {
                for (unsigned int x = 0; x < nImgPixX; ++x)
                {
                    inp0->data()(i,y,x,j) = ((x%2)==0) && ((y%2)==1);
                }
            }
        }
    }
    {
        ptr_t func		               = boost::make_shared<ResponseNormalization>(inp0->result(), 3, 0.0000125f, 0.5f);
        derivative_tester(*func,0,true);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_contrast_normalization){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 16;      // must be divisible by 16, it seems
    unsigned int nImgPixY  = 16;
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
    //inp0->data() = 1.f;
    for(unsigned int i=0; i<nImgChan; i++){
        for (unsigned int j = 0; j < nImg; ++j)
        {
            for (unsigned int y = 0; y < nImgPixY; ++y)
            {
                for (unsigned int x = 0; x < nImgPixX; ++x)
                {
                    //inp0->data()(i,y,x,j) = ((x%2)==0) && ((y%2)==1);
                    inp0->data()(i,y,x,j) = 0.1f + 0.9f * drand48();
                }
            }
        }
    }
    {
        ptr_t func		               = boost::make_shared<ContrastNormalization>(inp0->result(), 4, 0.0000125f, 0.5f);
        // TODO this function seems to have an unusually high error here.
        // Alex says he cannot tell why...!?
        derivative_tester(*func, 0, false, 0.08, 0,0);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_convolve_reorder){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 1;      // must be divisible by nGroups
    unsigned int nImgPixY  = 16;
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;
    //unsigned int nGroups  = 1;      // must be divisible by 2 ??

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX]);
    {
        ptr_t func		               = boost::make_shared<ReorderForConv>(inp0->result());
        derivative_tester(*func);
    }

    {
        ptr_t func		               = boost::make_shared<ReorderFromConv>(inp1->result());
        derivative_tester(*func);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_convolve_pooling){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 16;      // must be multiple of 16 for bprop
    unsigned int nImgPixX  = 16;
    unsigned int nImgPixY  = 16;
    unsigned int nImg     = 1;

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg]);
        ptr_t func		               = boost::make_shared<LocalPooling>(inp0->result(), 2, 2, cuv::alex_conv::PT_AVG);

        derivative_tester(*func);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_flatten){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result());

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 1);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8*3*3);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 2);

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
        BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3*3);

        derivative_tester(*func);
    }

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3][2]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 3);

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
        BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(2), 3*2);

        derivative_tester(*func);
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_reshape){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                     = boost::make_shared<Reshape>(inp0->result(), cuv::extents[3][8][3]);

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(1), 8);
        BOOST_CHECK_EQUAL(func->result()->shape.at(2), 3);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t rshp                     = boost::make_shared<Reshape>(inp0->result(), cuv::extents[3][-1]);
        ptr_t func                    = boost::make_shared<Pow>(2,rshp->result());

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Reshape>(inp0->result(), cuv::extents[8*3][-1]);

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8*3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);
    }

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3][2]);
        ptr_t func                       = boost::make_shared<Reshape>(inp0->result(), cuv::extents[8][-1][3]);

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
        BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3*2);
        BOOST_CHECK_EQUAL(func->result()->shape.at(2), 3);
    }
}


BOOST_AUTO_TEST_CASE(derivative_test_subtensor){
    typedef boost::shared_ptr<Op> ptr_t;
    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
       ptr_t func                     = boost::make_shared<Subtensor>(inp0->result(), cuv::indices[cuv::index_range(0,2)][cuv::index_range()]);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 2);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);

       derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
      ptr_t func                     = boost::make_shared<Subtensor>(inp0->result(), cuv::indices[2][cuv::index_range(0,2)][cuv::index_range()]);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);

      derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
      ptr_t func                     = boost::make_shared<Subtensor>(inp0->result(), cuv::indices[2][cuv::index_range(1,-1)][cuv::index_range()]);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 1);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);

      derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
      ptr_t func                     = boost::make_shared<Subtensor>(inp0->result(), cuv::indices[cuv::index_range(-5,-2)][cuv::index_range(1,-1)][cuv::index_range()]);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 3);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1), 1);
      BOOST_CHECK_EQUAL(func->result()->shape.at(2), 3);

      // copying of arbitrary strides not implemented (yet)
      //derivative_tester(*func);
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_concatenate){
    typedef boost::shared_ptr<Op> ptr_t;
    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
       boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
       ptr_t func                     = concatenate(inp0, inp1, 1);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1),6);

        inp0->data() = 0.2f;
        inp1->data() = 0.3f;
        function f(func, 0);
        matrix m = f.evaluate();

        for (unsigned int i = 0; i < func->result()->shape.at(0); ++i)
        {
            for (unsigned int j = 0; j < func->result()->shape.at(1); ++j)
            {
                if(j < 3){
                    BOOST_CHECK_EQUAL(m(i,j), 0.2f);
                }else{
                    BOOST_CHECK_EQUAL(m(i,j), 0.3f);
                }
            }
        }
        derivative_tester(*func);
    }

    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[15][4]);
      boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[15][4]);
      ptr_t func                     = concatenate(inp0, inp1, 0);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 30);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1),4);

       inp0->data() = 1;
       inp1->data() = 2;
       function f(func, 0);
       matrix m = f.evaluate();

       for (unsigned int i = 0; i < 30; ++i)
       {
           for (unsigned int j = 0; j < 4; ++j)
           {
               if(i < 15){
                   BOOST_CHECK_EQUAL(m(i,j), 1);
               }else{
                   BOOST_CHECK_EQUAL(m(i,j), 2);
               }
           }
       }
       derivative_tester(*func);
    }

    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][5]);
       boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
       ptr_t func                     = concatenate(inp0, inp1, 1);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1),8);

        inp0->data() = 0.2f;
        inp1->data() = 0.3f;
        function f(func, 0);
        matrix m = f.evaluate();

        for (unsigned int i = 0; i < func->result()->shape.at(0); ++i)
        {
            for (unsigned int j = 0; j < func->result()->shape.at(1); ++j)
            {
                if(j < 5){
                    BOOST_CHECK_EQUAL(m(i,j), 0.2f);
                }else{
                    BOOST_CHECK_EQUAL(m(i,j), 0.3f);
                }
            }
        }
        derivative_tester(*func);
    }
    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);
       boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[7][5]);
       ptr_t func                     = concatenate(inp0, inp1, 0);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 11);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1),5);

        inp0->data() = 0.2f;
        inp1->data() = 0.3f;
        function f(func, 0);
        matrix m = f.evaluate();

        for (unsigned int i = 0; i < func->result()->shape.at(0); ++i)
        {
            for (unsigned int j = 0; j < func->result()->shape.at(1); ++j)
            {
                if(i < 4){
                    BOOST_CHECK_EQUAL(m(i,j), 0.2f);
                }else{
                    BOOST_CHECK_EQUAL(m(i,j), 0.3f);
                }
            }
        }
        derivative_tester(*func);
    }

    // 3 dim case

    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
       boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
       ptr_t func                     = concatenate(inp0, inp1, 0);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1),5);
       BOOST_CHECK_EQUAL(func->result()->shape.at(2),8);

        inp0->data() = 0.2f;
        inp1->data() = 0.3f;
        function f(func, 0);
        matrix m = f.evaluate();

        for (unsigned int i = 0; i < func->result()->shape.at(0); ++i)
        {
            for (unsigned int j = 0; j < func->result()->shape.at(1); ++j)
            {
                for (unsigned int k = 0; k < func->result()->shape.at(2); ++k)
                {
                    if(i < 4){
                        BOOST_CHECK_EQUAL(m(i,j,k), 0.2f);
                    }else{
                        BOOST_CHECK_EQUAL(m(i,j,k), 0.3f);
                    }
                }
            }
        }
        derivative_tester(*func);
    }
    {
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
       boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
       ptr_t func                     = concatenate(inp0, inp1, 2);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 4);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1),5);
       BOOST_CHECK_EQUAL(func->result()->shape.at(2),16);

       derivative_tester(*func);
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[6][5][8]);
      boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
      ptr_t func                     = concatenate(inp0, inp1, 0);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 10);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1),5);
      BOOST_CHECK_EQUAL(func->result()->shape.at(2),8);

      derivative_tester(*func);
    }
}

/*
BOOST_AUTO_TEST_CASE(derivative_test_sumPooling){
   using namespace cuv;
   using namespace cuvnet;
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[6][6][6][5]);

   int nPoolFlt = 3;
   int nStride = 2;
   ptr_t pool = boost::make_shared<LocalPooling>(inp->result(), nPoolFlt , nStride , alex_conv::PT_SUM );
   derivative_testing::derivative_tester(*pool);
      cuv::safeThreadSync();
}
*/

BOOST_AUTO_TEST_CASE( sum_out_dim_test_first_dim )
{  
            unsigned int x = 21;
            unsigned int y = 5;
            unsigned int z = 9;
            unsigned int p = 17;

            using namespace cuv;
            using namespace cuvnet;
            typedef boost::shared_ptr<Op> op_ptr;
            
            //generate all inputs and fill them with rand vals
            boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][p]);
            fill_rnd_uniform(inp->data());
        
            op_ptr op = sum(inp, 0);
            // assumption: op has only one result
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            op->visit(pcv);
            BOOST_CHECK(pcv.plist.size()>0);

            std::vector<Op*> params(1);
            params[0] = inp.get();
            
            swiper swipe(*op, 0, params);

            swipe.fprop();
            cuvAssert(!cuv::has_nan(out_op->cdata()));
            cuvAssert(!cuv::has_inf(out_op->cdata())); 
            
            std::vector<unsigned int> desired_shape = {1, y, z, p};
            cuvAssert(out_op->cdata().shape() == desired_shape);
            
                    for (unsigned int j = 0; j < y; j++){
                        for (unsigned int k = 0; k < z; k++){
                            for (unsigned int l = 0; l < p; l++){
                                float  a = 0; 
                                for ( unsigned int i = 0; i < x; i++) a += inp->data()[indices[i][j][k]][l];
                                float  b = out_op->cdata()[indices[0][j][k]][l];
                                BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                            }
                        }
                    }
            derivative_testing::derivative_tester(*op, 0, true);
            cuv::safeThreadSync();
}


BOOST_AUTO_TEST_CASE( sum_out_dim_test_first_dim_mean )
{  
            unsigned int x = 3;
            unsigned int y = 5;
            unsigned int z = 9;
            unsigned int p = 5;

            using namespace cuv;
            using namespace cuvnet;
            typedef boost::shared_ptr<Op> op_ptr;
            
            //generate all inputs and fill them with rand vals
            boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][p]);
            fill_rnd_uniform(inp->data());
        
            op_ptr op = mean(inp, 0);
            // assumption: op has only one result
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            op->visit(pcv);
            BOOST_CHECK(pcv.plist.size()>0);

            std::vector<Op*> params(1);
            params[0] = inp.get();
            
            swiper swipe(*op, 0, params);

            swipe.fprop();
            cuvAssert(!cuv::has_nan(out_op->cdata()));
            cuvAssert(!cuv::has_inf(out_op->cdata())); 
            
            std::vector<unsigned int> desired_shape = {1, y, z, p};
            cuvAssert(out_op->cdata().shape() == desired_shape);
            
                    for (unsigned int j = 0; j < y; j++){
                        for (unsigned int k = 0; k < z; k++){
                            for (unsigned int l = 0; l < p; l++){
                                float  a = 0; 
                                for ( unsigned int i = 0; i < x; i++) a += inp->data()[indices[i][j][k]][l];
                                float  b = out_op->cdata()[indices[0][j][k]][l];
                                BOOST_CHECK_SMALL(fabs(a/(float)x - b) , 0.0001);  
                            }
                        }
                    }
            derivative_testing::derivative_tester(*op, 0, true);
            cuv::safeThreadSync();
}


BOOST_AUTO_TEST_CASE( sum_out_dim_test_last_dim )
{  
            unsigned int x = 21;
            unsigned int y = 5;
            unsigned int z = 9;
            unsigned int p = 17;

            using namespace cuv;
            using namespace cuvnet;
            typedef boost::shared_ptr<Op> op_ptr;
        
            //generate all inputs and fill them with rand vals
            boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][p]);
            fill_rnd_uniform(inp->data());

            op_ptr op = sum(inp, 3);
            // assumption: op has only one result
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            op->visit(pcv);
            BOOST_CHECK(pcv.plist.size()>0);

            std::vector<Op*> params(1);
            params[0] = inp.get();
            
            swiper swipe(*op, 0, params);

            swipe.fprop();
            cuvAssert(!cuv::has_nan(out_op->cdata()));
            cuvAssert(!cuv::has_inf(out_op->cdata())); 
            
            std::vector<unsigned int> desired_shape = {x, y, z, 1};
            cuvAssert(out_op->cdata().shape() == desired_shape);
            
                    for (unsigned int j = 0; j < x; j++){
                        for (unsigned int k = 0; k < y; k++){
                            for (unsigned int l = 0; l < z; l++){
                                float  a = 0; 
                                for ( unsigned int i = 0; i < p; i++) a += inp->data()[indices[j][k][l]][i];
                                float  b = out_op->cdata()[indices[j][k][l]][0];
                                BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                            }
                        }
                    }
            derivative_testing::derivative_tester(*op, 0, true);
            cuv::safeThreadSync();
}


BOOST_AUTO_TEST_CASE( sum_out_dim_test_last_dim )
{  
            unsigned int x = 21;
            unsigned int y = 5;
            unsigned int z = 9;
            unsigned int p = 17;

            using namespace cuv;
            using namespace cuvnet;
            typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
            typedef boost::shared_ptr<Op> op_ptr;
        
            //generate all inputs and fill them with rand vals
            boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][p]);
            fill_rnd_uniform(inp->data());

            op_ptr op = sum(inp, 3);
            // assumption: op has only one result
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            op->visit(pcv);
            BOOST_CHECK(pcv.plist.size()>0);

            std::vector<Op*> params(1);
            params[0] = inp.get();
            
            swiper swipe(*op, 0, params);

            swipe.fprop();
            cuvAssert(!cuv::has_nan(out_op->cdata()));
            cuvAssert(!cuv::has_inf(out_op->cdata())); 
            
            std::vector<unsigned int> desired_shape = {x, y, z, 1};
            cuvAssert(out_op->cdata().shape() == desired_shape);
            
                    for (unsigned int j = 0; j < x; j++){
                        for (unsigned int k = 0; k < y; k++){
                            for (unsigned int l = 0; l < z; l++){
                                float  a = 0; 
                                for ( unsigned int i = 0; i < p; i++) a += inp->data()[indices[j][k][l]][i];
                                float  b = out_op->cdata()[indices[j][k][l]][0];
                                BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                            }
                        }
                    }
            derivative_testing::derivative_tester(*op, 0, true);
            cuv::safeThreadSync();
}


BOOST_AUTO_TEST_CASE( Concatenate_first_dim )
{           
            unsigned int z = 13;
            unsigned int n = 3;
            unsigned int x = 2;
            unsigned int y = 5;
            using namespace cuv;
            using namespace cuvnet;
            typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
            typedef boost::shared_ptr<Op> op_ptr;
            
            std::vector< op_ptr >  input(n);       
            boost::shared_ptr<ParameterInput> in1;
            boost::shared_ptr<ParameterInput> in2;
            boost::shared_ptr<ParameterInput> in3;
            
        //generate all inputs and fill them with rand vals
        for ( unsigned int i = 0; i < n; i++){
                boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[x][y][z]);
                fill_rnd_uniform(inp->data());
                input[i] = inp;
                if ( i == 0 ) in1 = inp;
                else if (i==1) in2 = inp;
                else in3 = inp;
        }
        
            op_ptr op = concatenate(input, 0);
            // assumption: op has only one result
        boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

        // tell that we want derivative w.r.t. all params
        param_collector_visitor pcv;
        op->visit(pcv);
        BOOST_CHECK(pcv.plist.size()>0);

        std::vector<Op*> params(3);
        params[0] = in1.get();
        params[1] = in2.get();
        params[2] = in3.get();
        
        swiper swipe(*op, 0, params);

        swipe.fprop();
        cuvAssert(!cuv::has_nan(out_op->cdata()));
        cuvAssert(!cuv::has_inf(out_op->cdata()));
        
        
        std::vector<unsigned int> desired_shape = {3*x, y, z};
        cuvAssert(out_op->cdata().shape() == desired_shape);
            for ( unsigned int i = 0; i < n; i++){
                for (unsigned int j = 0; j < x; j++){
                    for (unsigned int k = 0; k < y; k++){
                        for (unsigned int l = 0; l < z; l++){
                            float  a = -10000;
                            if ( i == 0)  a = in1->data()[indices[j][k]][l];
                            if ( i == 1)  a = in2->data()[indices[j][k]][l];
                            if ( i == 2)  a = in3->data()[indices[j][k]][l];
                            
                            float  b = out_op->cdata()[indices[ i*x +j][k]][l];
                            BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                        }
                    }
                }
            }
            
        boost::shared_ptr<ParameterInput> delta = boost::make_shared<ParameterInput>(cuv::extents[3*x][y][z]);
        cuv::fill_rnd_uniform(delta->data());
        Op::result_t& r = op->result(0);
        r->delta.reset(new cuvnet::matrix(r->shape));
        r->delta.data() =  delta->data().copy();              
        
        swipe.bprop(false);
            for ( unsigned int i = 0; i < n; i++){
                for (unsigned int j = 0; j < x; j++){
                    for (unsigned int k = 0; k < y; k++){
                        for (unsigned int l = 0; l < z; l++){
                            float  a = -10000;
                            if ( i == 0)  a = in1->delta()[indices[j][k]][l];
                            if ( i == 1)  a = in2->delta()[indices[j][k]][l];
                            if ( i == 2)  a = in3->delta()[indices[j][k]][l];

                            float  b = delta->data()[indices[ i*x +j][k]][l];
                            BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                        }
                    }
                }
            }
            derivative_testing::derivative_tester(*op, 0, true);
            cuv::safeThreadSync();
}




BOOST_AUTO_TEST_CASE( Concatenate_old_interface )
{  
            unsigned int n = 2;
            unsigned int x = 2;
            unsigned int y = 5;
            unsigned int z = 9;

            using namespace cuv;
            using namespace cuvnet;
            typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
            typedef boost::shared_ptr<Op> op_ptr;
            
            boost::shared_ptr<ParameterInput> in1;
            boost::shared_ptr<ParameterInput> in2;
            
        //generate all inputs and fill them with rand vals
        for ( unsigned int i = 0; i < n; i++){
                boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[x][y][z]);
                fill_rnd_uniform(inp->data());
                if ( i == 0 ) in1 = inp;
                else if (i==1) in2 = inp;
        }
        
            op_ptr op = concatenate(in1, in2, 0);
            // assumption: op has only one result
        boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

        // tell that we want derivative w.r.t. all params
        param_collector_visitor pcv;
        op->visit(pcv);
        BOOST_CHECK(pcv.plist.size()>0);

        std::vector<Op*> params(2);
        params[0] = in1.get();
        params[1] = in2.get();
        
       // std::cout << "swiper.fprop()" << std::endl;
        swiper swipe(*op, 0, params);
        //    std::cout << "fprop.."<< std::endl;

        swipe.fprop();
        cuvAssert(!cuv::has_nan(out_op->cdata()));
        cuvAssert(!cuv::has_inf(out_op->cdata()));
        
      //  std::cout << "checking shapes"<< std::endl;
        std::vector<unsigned int> desired_shape = {2*x, y, z};
        cuvAssert(out_op->cdata().shape() == desired_shape);
      //  std::cout << "checking results"<< std::endl;
            for ( unsigned int i = 0; i < n; i++){
                for (unsigned int j = 0; j < x; j++){
                    for (unsigned int k = 0; k < y; k++){
                        for (unsigned int l = 0; l < z; l++){
                            float  a = -10000;
                            if ( i == 0)  a = in1->data()[indices[j][k]][l];
                            if ( i == 1)  a = in2->data()[indices[j][k]][l];
                            float  b = out_op->cdata()[indices[ i*x +j][k]][l];
                            BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                        }
                    }
                }
            }
        derivative_testing::derivative_tester(*op, 0, true);
        cuv::safeThreadSync();
}



BOOST_AUTO_TEST_CASE( Concatenate_N_last_dim )
{  
            unsigned int n = 3;
            unsigned int x = 2;
            unsigned int y = 5;
            unsigned int z = 2;
            unsigned int z1 = 3;

            using namespace cuv;
            using namespace cuvnet;
            typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
            typedef boost::shared_ptr<Op> op_ptr;
            
            std::vector< op_ptr >  input(n);
            boost::shared_ptr<ParameterInput> in1 = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][z1]);
            boost::shared_ptr<ParameterInput> in2 = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][z1]);
            boost::shared_ptr<ParameterInput> in3 = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][z1]);
            fill_rnd_uniform(in1->data());
            fill_rnd_uniform(in2->data());
            fill_rnd_uniform(in3->data());
            
            input[0] = in1;
            input[1] = in2;
            input[2] = in3;
            
            op_ptr op = concatenate(input, 3);
            // assumption: op has only one result
        boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

        // tell that we want derivative w.r.t. all params
        param_collector_visitor pcv;
        op->visit(pcv);
        BOOST_CHECK(pcv.plist.size()>0);

        std::vector<Op*> params(3);
        params[0] = in1.get();
        params[1] = in2.get();
        params[2] = in3.get();
        
        swiper swipe(*op, 0, params);
        swipe.fprop();
        cuvAssert(!cuv::has_nan(out_op->cdata()));
        cuvAssert(!cuv::has_inf(out_op->cdata())); 

        std::vector<unsigned int> desired_shape = {x, y, z, 3*z1};
        cuvAssert(out_op->cdata().shape() == desired_shape);        
        
        for ( unsigned int j = 0; j < x; j++)
            for ( unsigned int k = 0; k < y; k++)                
                for ( unsigned int h = 0; h < z; h++)
                    for ( unsigned int i = 0; i < n; i++)
                        for (unsigned int l = 0; l < z1; l++){
                            float  a = -10000;
                            if ( i == 0)  a = in1->data()[indices[j][k][h]][l];
                            if ( i == 1)  a = in2->data()[indices[j][k][h]][l];
                            if ( i == 2)  a = in3->data()[indices[j][k][h]][l];
                            unsigned int ind = i*z1 + l;
                            float  b = out_op->cdata()[indices[j][k][h]][ind];
                            BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                        }
                            
      boost::shared_ptr<ParameterInput> delta = boost::make_shared<ParameterInput>(cuv::extents[x][y][z][3*z1]);
        cuv::fill_rnd_uniform(delta->data());
        Op::result_t& r = op->result(0);
        r->delta.reset(new cuvnet::matrix(r->shape));
        r->delta.data() =  delta->data().copy();              
            
       
        swipe.bprop(false);
        for ( unsigned int j = 0; j < x; j++)
            for ( unsigned int k = 0; k < y; k++)                
                for ( unsigned int h = 0; h < z; h++)
                    for ( unsigned int i = 0; i < n; i++)
                        for (unsigned int l = 0; l < z1; l++){
                            float  a = -10000;
                            if ( i == 0)  a = in1->delta()[indices[j][k][h]][l];
                            if ( i == 1)  a = in2->delta()[indices[j][k][h]][l];
                            if ( i == 2)  a = in3->delta()[indices[j][k][h]][l];

                            float  b = delta->data()[indices[j][k][h]][i*z1 + l];
                            BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                        }        
            derivative_testing::derivative_tester(*op, 0, true);
cuv::safeThreadSync();
}



static float logAddExp_test(float t, float u){
        const float diff = (float)t - (float) u;
        if(diff > 0)
            return t + log1pf(expf(-diff));
        else if(diff<=0)
            return u + log1pf(expf(diff));
        else
            return t+u;
}


BOOST_AUTO_TEST_CASE(LOGADDEXP){
   using namespace cuv;
   using namespace cuvnet;
   typedef boost::shared_ptr<Op> ptr_t;
   unsigned int size = 23;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[size]);
   boost::shared_ptr<ParameterInput>  inp_s = boost::make_shared<ParameterInput>(cuv::extents[size]);
   boost::shared_ptr<ParameterInput>  delta = boost::make_shared<ParameterInput>(cuv::extents[size]);
   
   float a = 0.23f;
   cuv::fill_rnd_uniform(inp->data());
   inp_s->data() = inp->data().copy();

   ptr_t op = log_add_exp(inp, a);
   boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

   // tell that we want derivative w.r.t. all params
   param_collector_visitor pcv;
   op->visit(pcv);
   BOOST_CHECK(pcv.plist.size()>0);

   std::vector<Op*> params(1);
   params[0] = inp.get();

  // std::cout << "swiper.fprop()" << std::endl;
   swiper swipe(*op, 0, params);
   swipe.fprop();
   cuvAssert(!cuv::has_nan(out_op->cdata()));
   cuvAssert(!cuv::has_inf(out_op->cdata()));

   for (unsigned int i = 0; i < size; i++){
       float b = out_op->cdata()[i];
       float tmp = logAddExp_test(inp->data()[i], a);
       BOOST_CHECK_SMALL(fabs(tmp-b) , 0.001);       
    }

    //fill delta with data 
    Op::result_t& r = op->result(0);
    r->delta.reset(new cuvnet::matrix(r->shape));
    fill_rnd_uniform(r->delta.data());   
    delta->data() = r->delta.data().copy();

    //save input data..
    
    swipe.bprop(false );
      for (unsigned int i = 0; i < size; i++){
      float tmp =  expf(inp_s->data()[i]);
      tmp /= (tmp + expf(a));
      tmp *= delta->data()[i];
      float b = inp->delta()[i];
      float diff = fabs(tmp-b);
      BOOST_CHECK_SMALL(diff , 0.001f);
   }
   derivative_testing::derivative_tester(*op);
   cuv::safeThreadSync();    
}

#ifndef NO_THEANO_WRAPPERS


BOOST_AUTO_TEST_CASE(theano_flip_dims){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::theano_ops;

        {
            unsigned int nImgChan = 3;      // must be divisible by nGroups
            unsigned int nImgPixX = 5;
            unsigned int nImgPixY = 5;
            unsigned int nImg     = 2;


            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX], "inputs");

                {
                    ptr_t func                       = boost::make_shared<FlipDims>(inp0->result(), cuv::extents[0][0][1][1]);
                    derivative_tester(*func,0,false,.03f);
                }
            }
        }
}
BOOST_AUTO_TEST_CASE(theano_convolve){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::theano_conv;

        {
            unsigned int nImgChan = 3;      // must be divisible by nGroups
            unsigned int nImgPixX = 5;
            unsigned int nImgPixY = 5;
            unsigned int nImg     = 1;

            unsigned int nFiltChan = nImgChan;
            unsigned int nFiltPixX  = 3;
            unsigned int nFiltPixY  = 3;
            unsigned int nFilt     = 2;

            {
               boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX], "inputs");
               boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFilt][nFiltChan][nFiltPixY][nFiltPixX], "weights");
               boost::shared_ptr<ParameterInput> padding_bias = boost::make_shared<ParameterInput>(cuv::extents[nFilt][nFiltPixY + nImgPixY - 1][nFiltPixX + nImgPixX - 1], "padding_bias");

               {
                 ptr_t func                       = boost::make_shared<Convolve2dTheano>(inp0->result(), inp1->result(), "valid");
                 derivative_tester(*func,0,false,.03f);
               }
               {
                  ptr_t func                       = boost::make_shared<Convolve2dTheano>(inp0->result(), inp1->result(), "full");
                  derivative_tester(*func,0,false,.03f);
               }
               {
                 ptr_t func                       = boost::make_shared<Convolve2dTheano>(inp0->result(), inp1->result(), padding_bias->result(), "full");
                 derivative_tester(*func,0,false,.03f);
               }
            }
        }

}

BOOST_AUTO_TEST_CASE(theano_shuffle_dim){
    typedef boost::shared_ptr<Op> ptr_t;
    using namespace cuv::theano_ops;
        {
            unsigned int nImgChan = 3;      // must be divisible by nGroups
            unsigned int nImgPixX = 5;
            unsigned int nImgPixY = 5;
            unsigned int nImg     = 2;


            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX], "inputs");

                {
                    ptr_t func                       = boost::make_shared<ShuffleDim>(inp0->result(), cuv::extents[1][0][2][3]);
                    derivative_tester(*func,0,false,.03f);
                }
                {
                    ptr_t func                       = boost::make_shared<ShuffleDim>(inp0->result(), cuv::extents[0][1][3][2]);
                    derivative_tester(*func,0,false,.03f);
                }
            }
        }
}

#endif


BOOST_AUTO_TEST_SUITE_END()




