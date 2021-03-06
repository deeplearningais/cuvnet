#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <boost/assign.hpp>

#define CUVNET_PRECISE_SUM 1

#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <cuvnet/tools/function.hpp>
#include <cuvnet/tools/matwrite.hpp>
#include <cuvnet/tools/logging.hpp>

#include <cuv/tools/timing.hpp>

#include <cuvnet/ops.hpp>
#include <cuvnet/datasets/detection.hpp>

#include <boost/test/unit_test.hpp>
#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#define EXPECT_NEAR(X,Y,D) BOOST_REQUIRE_LT(((X)-(Y))*((X)-(Y)), ((D)*(D)))
#endif

using namespace cuvnet;
using std::printf;
using namespace cuvnet::derivative_testing;

namespace{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("derivative_test"));
}

struct Fix{
    boost::shared_ptr<Tracer> m_suite_trace;
    boost::shared_ptr<Tracer> m_test_trace;
    Fix(){
        //m_suite_trace.reset(new Tracer(g_log, boost::unit_test::framework::current_test_case().p_parent_id.));
        m_test_trace.reset(new Tracer(g_log, boost::unit_test::framework::current_test_case().p_name));
    }
};

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
            cuv::safeThreadSync();               \
		}                                       \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

BOOST_FIXTURE_TEST_SUITE( op_test, Fix )

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

BOOST_FIXTURE_TEST_SUITE( derivative_test, Fix )

void fill_with_permuted_sequence(matrix& m){
    cuv::sequence(m);
    cuv::tensor<float, cuv::host_memory_space> t = m;
    std::random_shuffle(t.ptr(), t.ptr() + t.size());
    m = t;
}

BOOST_AUTO_TEST_CASE(derivative_test_pipe){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Pipe>(inp->result(),0);
   derivative_tester (*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_rowrange_select){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);
   boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);

   ptr_t func                    = row_range_select(inp0,inp1,2,0); // select 2 of the rows (fix to 0-2 for testing)

   derivative_tester(*result(func,0)).test();
   derivative_tester(*result(func,1)).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_row_select){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);

   ptr_t func                    = row_select(inp0,inp1,1);

   derivative_tester(*result(func,0)).test();
   derivative_tester(*result(func,1)).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_scalar_like){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                     = boost::make_shared<ScalarLike>(inp->result(), 3.4f);
   derivative_tester(*func).test();
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

   derivative_tester(*pow).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_exp){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t exp                     = boost::make_shared<Exp>(2,inp->result());
   derivative_tester(*exp).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_abs){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Abs>(inp->result());
   derivative_tester(*func).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_log){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Log>(inp->result());
   derivative_tester(*func).values(0.1, 2.).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_mean){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Mean>(inp->result());
    derivative_tester(*func).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_tanh){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Tanh>(inp->result());
    derivative_tester(*func).verbose().test();
}
BOOST_AUTO_TEST_CASE(derivative_test_sin){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Sin>(inp->result());
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_cos){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Cos>(inp->result());
    derivative_tester(*func).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_add_scalar){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<AddScalar>(1.f,inp->result());
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_mult_scalar){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<MultScalar>(1.5f,inp->result());
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_subtract_from_scalar){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    //ptr_t func                    = boost::make_shared<SubtractFromScalar>(1.f,inp->result());
    ptr_t func                    = 1.f-inp;
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_rectified_linear){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    float epsilon = 1.0;
    for(unsigned int memopt=0; memopt<2; memopt++)
    {
        ptr_t func                    = boost::make_shared<RectifiedLinear>(inp->result(), memopt);
        LOG4CXX_WARN(g_log, "memopt:"<<memopt<<" positive");
        derivative_tester(*func).epsilon(epsilon).values(epsilon,2.*epsilon).reinit(memopt).test();
        LOG4CXX_WARN(g_log, "memopt:"<<memopt<<" negative");
        derivative_tester(*func).full_jacobian().epsilon(epsilon).values(-3.*epsilon,-2*epsilon).only_variant(1).reinit(memopt).test();
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_multiply){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Multiply>(inp0->result(), inp1->result());
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_atan2){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5], "y");
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5], "x");
    ptr_t func                     = boost::make_shared<Atan2>(inp0->result(), inp1->result());
    derivative_tester(*func).test();
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
    derivative_tester(*func).verbose().values(0., 1.).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_hinge_loss){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    inp0->set_derivable(false);
    ptr_t func                     = boost::make_shared<HingeLoss>(inp0->result(), inp1->result(), false);
    derivative_tester(*func).verbose().test();
}
BOOST_AUTO_TEST_CASE(derivative_test_squared_hinge_loss){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    inp0->set_derivable(false);
    ptr_t func                     = boost::make_shared<HingeLoss>(inp0->result(), inp1->result(), true);
    derivative_tester(*func).verbose().test();
}
BOOST_AUTO_TEST_CASE(derivative_test_neg_cross_entropy_logistic){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5]);
    ptr_t func                     = boost::make_shared<NegCrossEntropyOfLogistic>(inp0->result(), inp1->result());
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(bernoulli_kl){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5], "inp0");
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5], "inp1");
        inp0->set_derivable(false);
        ptr_t func                     = boost::make_shared<BernoulliKullbackLeibler>(inp0->result(), inp1->result());
        derivative_tester(*func).values(0.1, 0.9).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5], "inp0");
        ptr_t func                     = boost::make_shared<BernoulliKullbackLeibler>(0.5f, inp0->result());
        derivative_tester(*func).values(0.1, 0.9).test(); // test in the range of 0.1, 0.9
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_axpby){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.3, -2.5);
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_axpby_broadcast){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5], "inp0");
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[1], "scalar");
    ptr_t func                  = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.2, -2.6);
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_sum_mat_to_vec_squared){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        TRACE(g_log, "dim1of2");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0,false,true);
        derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim2of2");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,false,true);
        derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim3of4");
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
       ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2,false,true);
       derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim2of4");
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,false,true);
      derivative_tester(*func).test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_sum_mat_to_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        TRACE(g_log, "dim1of2");
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0);
      derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim2of2");
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1);
      derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim2of3");
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][4][5]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1, false, false);
      derivative_tester(*func).only_variant(4).test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mean_mat_to_vec_squared){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        TRACE(g_log, "dim1of2");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0,true,true);
        derivative_tester(*func).verbose().test();
    }
    {
        TRACE(g_log, "dim2of2");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true,true);
        derivative_tester(*func).verbose().test();
    }
    {
        TRACE(g_log, "dim3of4");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2,true,true);
        derivative_tester(*func).verbose().test();
    }
    {
        TRACE(g_log, "dim2of4");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true,true);
        derivative_tester(*func).verbose().test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mean_mat_to_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        TRACE(g_log, "dim1of2");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0,true);
        derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim2of2");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true);
        derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim3of4");
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
       ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2,true);
       derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim2of4");
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4][6]);
      ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),1,true);
      derivative_tester(*func).test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_sum_mat_to_vec3d){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        TRACE(g_log, "dim1of3");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),0);
        derivative_tester(*func).test();
    }
    {
        TRACE(g_log, "dim3of3");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][4]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),2);
        derivative_tester(*func).test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_noiser_dropout){
        cuv::initialize_mersenne_twister_seeds(12);
        typedef boost::shared_ptr<Noiser> ptr_t;
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[6]);
        for(int compensate=0; compensate<2; compensate++){
            ptr_t func                     = boost::make_shared<Noiser>(inp0->result(), 0.5, Noiser::NT_ZERO_OUT, compensate);
            LOG4CXX_WARN(g_log, "active: false" << "compensate: " << compensate);
            func->set_active(false);
            derivative_tester(*func).epsilon(1.).reinit(true).test();
            LOG4CXX_WARN(g_log, "active: true" << "compensate: " << compensate);
            func->set_active(true);
            derivative_tester(*func).epsilon(1.).reinit(true).test();
        }
}
BOOST_AUTO_TEST_CASE(derivative_test_noiser_salt_and_pepper){
        cuv::initialize_mersenne_twister_seeds();
        typedef boost::shared_ptr<Op> ptr_t;
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);
        inp0->set_derivable(false); // this function does not implement bprop
        {
            ptr_t func                     = boost::make_shared<Noiser>(inp0->result(), 0.5, Noiser::NT_SALT_AND_PEPPER, false);
            derivative_tester(*func).epsilon(1).reinit(true).test();
        }
}
BOOST_AUTO_TEST_CASE(derivative_test_noiser_normal){
        cuv::initialize_mersenne_twister_seeds();
        typedef boost::shared_ptr<Op> ptr_t;
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);
        {
            ptr_t func                     = boost::make_shared<Noiser>(inp0->result(), 0.5, Noiser::NT_NORMAL, false);
            derivative_tester(*func).epsilon(1).reinit(true).test();
        }
}
BOOST_AUTO_TEST_CASE(derivative_test_sum){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[2][2]);
    ptr_t func                     = boost::make_shared<Sum>(inp0->result());
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_transpose){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[2][3]);
    ptr_t func                     = boost::make_shared<Transpose>(inp0->result());
    derivative_tester(*func).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_prod){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5][3]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][8]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result());
        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][8]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t');
        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5][3]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'n','t');
        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
        ptr_t func0		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
        ptr_t func1		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
        ptr_t func 		       = boost::make_shared<Axpby>(func0->result(), func1->result(), 1.3,1.5);
        derivative_tester(*func).test();
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_prod_reshape){
    typedef boost::shared_ptr<Op> ptr_t;
    double eps = 1.0; // everything is linear!
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5][2][3]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][2][8]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result());
        determine_shapes(*func);
        BOOST_REQUIRE_EQUAL(func->result()->shape.size(), 4);
        BOOST_REQUIRE_EQUAL(func->result()->shape[0], 5);
        BOOST_REQUIRE_EQUAL(func->result()->shape[1], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[2], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[3], 8);
        derivative_tester(*func).epsilon(eps).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][2][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][2][8]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t');
        determine_shapes(*func);
        BOOST_REQUIRE_EQUAL(func->result()->shape.size(), 4);
        BOOST_REQUIRE_EQUAL(func->result()->shape[0], 5);
        BOOST_REQUIRE_EQUAL(func->result()->shape[1], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[2], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[3], 8);
        derivative_tester(*func).epsilon(eps).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5][2][3]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][2][3]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'n','t');
        determine_shapes(*func);
        BOOST_REQUIRE_EQUAL(func->result()->shape.size(), 4);
        BOOST_REQUIRE_EQUAL(func->result()->shape[0], 5);
        BOOST_REQUIRE_EQUAL(func->result()->shape[1], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[2], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[3], 8);
        derivative_tester(*func).epsilon(eps).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][2][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][2][3]);
        ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
        determine_shapes(*func);
        BOOST_REQUIRE_EQUAL(func->result()->shape.size(), 4);
        BOOST_REQUIRE_EQUAL(func->result()->shape[0], 5);
        BOOST_REQUIRE_EQUAL(func->result()->shape[1], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[2], 2);
        BOOST_REQUIRE_EQUAL(func->result()->shape[3], 8);
        derivative_tester(*func).epsilon(eps).test();
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_sq_axpby){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp0->result(), 1.5f, -1.8f);
    func                           = boost::make_shared<Pow>(2.f,func->result());
    derivative_tester(*func).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_xtx){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = inp0*inp0;
    derivative_tester(*func).test();
}
BOOST_AUTO_TEST_CASE(derivative_test_xt1mx){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = inp0*(1.f-inp0);
    derivative_tester(*func).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_add_to_param){
    typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t tmp0                     = boost::make_shared<Pow>(2.f, inp0->result());
    ptr_t tmp1                     = boost::make_shared<Pow>(3.f, inp0->result());
    ptr_t func                     = boost::make_shared<Sum>(tmp0->result());
    add_to_param(func, tmp1);  // sum(  x^2+x^3 )
    derivative_tester(*func).test();
}

BOOST_AUTO_TEST_CASE(derivative_test_softmax){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        TRACE(g_log, "dim0");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 0);
        derivative_tester(*func).full_jacobian().test();
    }
    {
        TRACE(g_log, "dim1");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 1);
        derivative_tester(*func).full_jacobian().test();
    }
    {
        TRACE(g_log, "dim1of4");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][3]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 0);
       
        cuv::fill_rnd_uniform(inp0->data());
        function f(func, 0);
        matrix res = f.evaluate();

        // checking fprop
        cuv::tensor<float, cuv::dev_memory_space> ref(cuv::extents[3][5][5][3]);
        for (int a = 0; a < 5; a++) {
            for (int b = 0; b < 5; b++) {
                for (int c = 0; c < 3; c++) {
                    double sum = 0;
                    for (int x = 0; x < 3; x++) {
                        double tmp = std::exp(inp0->data()(x,a,b,c));
                        ref(x,a,b,c) = tmp;
                        sum += tmp;
                    }
                    for (int x = 0; x < 3; x++) {
                        ref(x,a,b,c) /= sum;

                        BOOST_REQUIRE_CLOSE((float) ref(x,a,b,c), (float) res(x,a,b,c), 0.001);
                    }
                }
            }
        }
        
        derivative_tester(*func).full_jacobian().test();
    }
    {
        TRACE(g_log, "dim2of4");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][3]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 1);

        cuv::fill_rnd_uniform(inp0->data());
        function f(func, 0);
        matrix res = f.evaluate();

        // checking fprop 
        cuv::tensor<float, cuv::dev_memory_space> ref(cuv::extents[3][5][5][3]);
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 5; b++) {
                for (int c = 0; c < 5; c++) {
                    double sum = 0;
                    for (int x = 0; x < 3; x++) {
                        double tmp = std::exp(inp0->data()(a,b,c,x));
                        ref(a,b,c,x) = tmp;
                        sum += tmp;
                    }
                    for (int x = 0; x < 3; x++) {
                        ref(a,b,c,x) /= sum;

                        BOOST_REQUIRE_CLOSE((float) ref(a,b,c,x), (float) res(a,b,c,x), 0.001);
                    }
                }
            }
        }

        derivative_tester(*func).full_jacobian().test();
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_mll2){
    // subtracting the maximum for every row yields gradients which are
    // different to the analytical gradient, but still look OK-ish.
    typedef boost::shared_ptr<Op> ptr_t;
    {
        TRACE(g_log, "Axis0");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8]);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss2>(inp0->result(), inp1->result(), 0);
        cuv::fill_rnd_uniform(inp0->data());

        // fprop of MultinomialLogisticLoss2 subtracts maximum, but bprop does not consider this.
        cuv::tensor<float, cuv::dev_memory_space> vec(8);
        cuv::reduce_to_col(vec, inp0->data(), cuv::RF_MAX, -1.f, 0.f);
        cuv::matrix_plus_col(inp0->data(), vec);

        inp1->data() = 0.f;
        for(unsigned int i=0; i<inp0->data().shape(0); i++){
            int klass = (int) (inp0->data().shape(1) * drand48());
            inp1->data()(i) = klass;
        }
        //{
        //    cuvnet::function f(func, 1);
        //    tofile("sm.npy", f.evaluate());
        //    tofile("inp0.npy", inp0->data());
        //    tofile("inp1.npy", inp1->data());
        //}
        //{
        //    cuvnet::function f(func, 0);
        //    std::cout << "logprob: " << f.evaluate()[0] <<std::endl;
        //}
        //{
        //    cuvnet::delta_function f(func, func);
        //    tofile("d_sm.npy", f.evaluate());
        //    //tofile("inp0.npy", inp0->data());
        //    //tofile("inp1.npy", inp1->data());
        //}

        inp1->set_derivable(false);
        derivative_tester(*func).full_jacobian().values(0, 0).test();
    }
    {
        TRACE(g_log, "Axis1");
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5]);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss2>(inp0->result(), inp1->result(), 1);
        cuv::fill_rnd_uniform(inp0->data());
        
        // fprop of MultinomialLogisticLoss2 subtracts maximum, but bprop does not consider this.
        cuv::tensor<float, cuv::dev_memory_space> vec(5);
        cuv::reduce_to_row(vec, inp0->data(), cuv::RF_MAX, -1.f, 0.f);
        cuv::matrix_plus_row(inp0->data(), vec);

        inp1->data() = 0.f;
        for(unsigned int i=0; i<inp0->data().shape(1); i++){
            int klass = (int) (inp0->data().shape(0) * drand48());
            inp1->data()(i) = klass;
        }
        inp1->set_derivable(false);
        derivative_tester(*func).full_jacobian().values(0, 0).test();
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_mll){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 1);
        derivative_tester(*func).test();
    }
    // higher dimensional input
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 3);
        derivative_tester(*func).test();
    }

    ///// SoftMax result of MultinomialLogisticLoss
    if(0){
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func).result(1).test();
    }
    if(0){
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 1);
        derivative_tester(*func).result(1).test();
    }
    // higher dimensional input
    if(0){
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func).result(1).test();
    }
    if(0){
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5][5][4]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 3);
        derivative_tester(*func).result(1).test();
    }

}

BOOST_AUTO_TEST_CASE(derivative_test_mat_plus_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5], "mat");
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3], "vec");
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 0);

        derivative_tester(*func).epsilon(10.).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func).epsilon(10.).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][6]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func).epsilon(10.).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][6]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [2]);
        ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), 2);

        derivative_tester(*func).epsilon(10.).test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mat_times_vec){
    // requires less tight precision, but visually jacobians correspond well minus noise
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 0);

        derivative_tester(*func).epsilon(10.).precision(0.015).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func).epsilon(10.).precision(0.015).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func).epsilon(10.).precision(0.015).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [2]);
        ptr_t func		           = boost::make_shared<MatTimesVec>(inp0->result(), inp1->result(), 2);

        derivative_tester(*func).epsilon(10.).precision(0.015).test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_mat_div_vec){
    typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 0);

        derivative_tester(*func).values(0.2, 1.0).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func).values(0.2, 1.0).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [5]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 1);

        derivative_tester(*func).values(0.2, 1.0).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5][2][4]);
        boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents   [2]);
        ptr_t func		           = boost::make_shared<MatDivideVec>(inp0->result(), inp1->result(), 2);

        derivative_tester(*func).values(0.2, 1.0).test();
    }
}
BOOST_AUTO_TEST_CASE(derivative_test_convolve){
    typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;

    for (int padding = 0; padding < 2; ++padding)
    {
        {
            unsigned int nImgChan = 1;      // must be divisible by nGroups
            unsigned int nImgPixX = 6;
            unsigned int nImgPixY = 6;
            unsigned int nImg     = 4;
            unsigned int nGroups  = 1;      // must be divisible by 2 ??

            unsigned int nFiltChan = nImgChan/nGroups;
            unsigned int nFiltPixX  = 3;
            unsigned int nFilt     = 16;

            //unsigned int nResPix   = nImgPixX-nFiltPixX+1;

            {
                // sparse convolution
                unsigned int nGroups = 2;
                unsigned int nImgChan = 16;
                unsigned int nFiltChan = 8;
                unsigned int nFilt     = 16;
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt * nGroups], "weights");
                boost::shared_ptr<Convolve> func        = boost::make_shared<Convolve>(inp0->result(), inp1->result(), 
                        padding, padding, 1, nGroups, 0);
                func->set_random_sparse(nFiltChan);

                LOG4CXX_WARN(g_log, "in 1st case convolution, padding: " << padding);
                derivative_tester(*func).epsilon(1.0).test();
            }

            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt], "weights");
                ptr_t func                       = boost::make_shared<Convolve>(inp0->result(), inp1->result(), 
                        padding, padding, 1, nGroups, 1);

                // it might not be possible to derive for images if they have only 3 channels!
                if(nImgChan % 4 != 0)
                    inp0->set_derivable(false);

                LOG4CXX_WARN(g_log, "in 2nd case convolution, padding: " << padding);
                derivative_tester(*func).epsilon(1.0).test();
            }

            {
                unsigned int nImgChan = 16;      // must be divisible by nGroups
                unsigned int nFiltChan = nImgChan/nGroups;
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt], "weights");
                ptr_t func                       = boost::make_shared<Convolve>(inp0->result(), inp1->result(), 
                        padding, padding, 1, nGroups, 1);

                LOG4CXX_WARN(g_log, "in 3rd case convolution, padding: " << padding);
                derivative_tester(*func).epsilon(1.0).test();
            }
        }

        {
            // reconstruction of auto-encoder... go from many "images" to one "filter".
            // this does not work in a straight-forward way, since alex' convs only
            // support n*16 outputs.
            // the version used here will use (temporarily) more memory and will be slower
            // (than a hypothetical "optimal" version)
            unsigned int nImgChan = 1;      // must be divisible by nGroups
            unsigned int nImgPixY  = 6;
            unsigned int nImgPixX  = 6;
            unsigned int nImg     = 4;
            unsigned int nGroups  = 1;      // must be divisible by 2 ??

            unsigned int nFiltChan = nImgChan/nGroups;
            unsigned int nFiltPixX  = 3;
            unsigned int nFilt     = 1;

            //unsigned int nResPix   = nImgPixX-nFiltPixX+1;
            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg],"inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt],"weights");
                ptr_t func                     = boost::make_shared<Convolve>(inp0->result(), inp1->result(), padding, 0, 1, 1);

                LOG4CXX_WARN(g_log, "in 4th case convolution, padding: " << padding);
                derivative_tester(*func).epsilon(1.0).test();
            }
        }
    }
}





BOOST_AUTO_TEST_CASE(test_derivative_select_entry){
    typedef boost::shared_ptr<Op> ptr_t;

    boost::shared_ptr<ParameterInput>  mat = boost::make_shared<ParameterInput>(cuv::extents[20][10], "mat");
    boost::shared_ptr<ParameterInput>  vec = boost::make_shared<ParameterInput>(cuv::extents[20], "vec");
    vec->set_derivable(false);

    cuv::fill_rnd_uniform(mat->data());
    for(unsigned int i=0; i< 20; i++){
        vec->data()[i] = (float)(int)(drand48()*10.0);
    }

    {
        ptr_t func   = boost::make_shared<RemoveEntryInEveryRow>(mat->result(), vec->result());
        derivative_tester dt(*func);
        dt.precision(0.03).epsilon(1).values(0, 0).full_jacobian().test();
    }
    {
        ptr_t func   = boost::make_shared<SelectEntryInEveryRow>(mat->result(), vec->result());
        derivative_tester dt(*func);
        dt.precision(0.03).epsilon(1).values(0, 0).full_jacobian().test();
    }
}

void test_derivative_test_tuple_ops(cuv::alex_conv::tuplewise_op_functor to){
    typedef boost::shared_ptr<Op> ptr_t;

    bool reinit = to != cuv::alex_conv::TO_MAX;
    using namespace cuv::alex_conv;

    {
       LOG4CXX_WARN(g_log, "in first case tuplewise op");
       unsigned int sub_size = 3;
       unsigned int nImgChan = 2 * sub_size;      // must be divisible by nGroups
       unsigned int nImgPixX = 2;
       unsigned int nImgPixY = 2;
       unsigned int nImg     = 2;

       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
       ptr_t func   = boost::make_shared<Tuplewise_op>(inp0->result(), 0, sub_size, to, 0.0001f);

       fill_with_permuted_sequence(inp0->data());
       if(to != cuv::alex_conv::TO_MAX){
           inp0->data() *= 4.f / cuv::maximum(inp0->data());
           inp0->data() -= 2.f;
       }
    
       derivative_tester dt(*func);
       if(reinit)
           dt.test();
       else
           dt.values(0, 0).epsilon(0.1).test();
    }



    {
        LOG4CXX_WARN(g_log, "in 2nd case tuplewise op" );
        unsigned int sub_size = 3;
        unsigned int nImgChan = 2 * sub_size;      // must be divisible by nGroups
        unsigned int nImgPixX = 2;
        unsigned int nImgPixY = 2;
        unsigned int nImg     = 2;

        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgPixY][nImgPixX][nImg][nImgChan], "inputs");
        ptr_t func   = boost::make_shared<Tuplewise_op>(inp0->result(), 3, sub_size, to, 0.0001f);

        fill_with_permuted_sequence(inp0->data());
        if(to != cuv::alex_conv::TO_MAX){
            inp0->data() *= 0.1f / cuv::maximum(inp0->data());
            inp0->data() -= 0.05f;
        }

        derivative_tester dt(*func);
       if(reinit)
           dt.test();
       else
           dt.values(0, 0).epsilon(0.1).test();
    }

    {
        LOG4CXX_WARN(g_log, "in 3rd case tuplewise op" );
        unsigned int sub_size = 3;
        unsigned int nImgChan = 2 * sub_size;      // must be divisible by nGroups
        unsigned int nImg     = 8;

        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan], "inputs");
        ptr_t func   = boost::make_shared<Tuplewise_op>(inp0->result(), 1, sub_size, to, 0.0001f);

        fill_with_permuted_sequence(inp0->data());
        if(to != cuv::alex_conv::TO_MAX){
            inp0->data() *= 4.f / cuv::maximum(inp0->data());
            inp0->data() -= 2.f;
        }
        derivative_tester dt(*func);
       if(reinit)
           dt.test();
       else
           dt.values(0, 0).epsilon(0.1).test();
    }

}
BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_norm){
    test_derivative_test_tuple_ops(cuv::alex_conv::TO_NORM);
}

BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_max){
    test_derivative_test_tuple_ops(cuv::alex_conv::TO_MAX);
}

//BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_mean){
    //test_derivative_test_tuple_ops(cuv::alex_conv::TO_MEAN);
//}
//BOOST_AUTO_TEST_CASE(derivative_test_tuplewise_subsample){
//    test_derivative_test_tuple_ops(cuv::alex_conv::TO_SUBSAMPLE);
//}
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
        derivative_tester(*func).test();
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
        derivative_tester(*func).test();
    }
    kernel = 0.f;
    kernel[kernel.size()/2] = 1.f;
    {
        ptr_t func		               = boost::make_shared<SeparableFilter>(inp0->result(), kernel);
        derivative_tester(*func).test();
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
            derivative_tester(*func).test();
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
 *        derivative_tester(*func).test();
 *    }
 *}
 */


BOOST_AUTO_TEST_CASE(response_normalization_cross_maps_caffe){
	typedef boost::shared_ptr<Op> ptr_t;

    unsigned int nImgChan = 16;
    unsigned int nImgPixY  = 16;
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX]);

    float alpha = 0.0000125f;
    float beta = 0.75f;
    float size = 3;
    {
        ptr_t op = boost::make_shared<ResponseNormalizationAcrossMapsCaffe>(inp0->result(), size, alpha, beta);
        derivative_tester(*op).test();

    }
}

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
        //derivative_tester(*func).verbose().test();
        
        LOG4CXX_WARN(g_log, "derivative_test_response_normalization_cross_maps crashes with memory access violation");
        BOOST_CHECK(0);
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
        //derivative_tester(*func).verbose(true).test();
       
        LOG4CXX_WARN(g_log, "derivative_test_response_normalization crashes with memory access violation");
        BOOST_CHECK(0);
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
        //derivative_tester(*func).verbose().test();
        
        LOG4CXX_WARN(g_log, "derivative_test_contrast_normalization crashes with memory access violation");
        BOOST_CHECK(0);
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
        derivative_tester(*func).test();
    }

    {
        ptr_t func		               = boost::make_shared<ReorderFromConv>(inp1->result());
        derivative_tester(*func).test();
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

        derivative_tester(*func).test();
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

        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 2);

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
        BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3*3);

        derivative_tester(*func).test();
    }

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3][2]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 3);

        determine_shapes(*func);
        BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
        BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);
        BOOST_CHECK_EQUAL(func->result()->shape.at(2), 3*2);

        derivative_tester(*func).test();
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

        derivative_tester(*func).test();
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t rshp                     = boost::make_shared<Reshape>(inp0->result(), cuv::extents[3][-1]);
        ptr_t func                    = boost::make_shared<Pow>(2,rshp->result());

        derivative_tester(*func).test();
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

       derivative_tester(*func).test();
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
      ptr_t func                     = boost::make_shared<Subtensor>(inp0->result(), cuv::indices[2][cuv::index_range(0,2)][cuv::index_range()]);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);

      derivative_tester(*func).test();
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
      ptr_t func                     = boost::make_shared<Subtensor>(inp0->result(), cuv::indices[2][cuv::index_range(1,-1)][cuv::index_range()]);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 1);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);

      derivative_tester(*func).test();
    }
    {
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][6]);
      ptr_t func                     = boost::make_shared<Subtensor>(inp0->result(), cuv::indices[cuv::index_range()][cuv::index_range(0,3)]);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 3);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1), 3);

      derivative_tester(*func).test();
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
      //derivative_tester(*func).test();
    }
}

BOOST_AUTO_TEST_CASE(derivative_test_concatenate){
    typedef boost::shared_ptr<Op> ptr_t;
    double epsilon = 1.0;
    {
       TRACE(g_log, "dim2along1a");
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
       boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
       ptr_t func                     = concatenate(inp0, inp1, 1);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 2);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 8);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1),6);

       int s = inp0->data().size();
       cuv::sequence(inp0->data());
       cuv::sequence(inp1->data());
       inp1->data() += (float) s;

        function f(func, 0);
        matrix m = f.evaluate();

        for (unsigned int i = 0; i < func->result()->shape.at(0); ++i)
        {
            for (unsigned int j = 0; j < func->result()->shape.at(1); ++j)
            {
                if(j < 3){
                    BOOST_CHECK_EQUAL(m(i,j), inp0->data()(i,j));
                }else{
                    BOOST_CHECK_EQUAL(m(i,j), inp1->data()(i,j-3));
                }
            }
        }
        derivative_tester(*func).epsilon(epsilon).test();
    }

    {
       TRACE(g_log, "dim2along0a");
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
       derivative_tester(*func).epsilon(epsilon).test();
    }

    {
       TRACE(g_log, "dim2along1b");
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
        derivative_tester(*func).epsilon(epsilon).test();
    }
    {
       TRACE(g_log, "dim2along0b");
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
        derivative_tester(*func).epsilon(epsilon).test();
    }

    // 3 dim case

    {
       TRACE(g_log, "dim3along0a");
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
        derivative_tester(*func).epsilon(epsilon).test();
    }
    {
       TRACE(g_log, "dim3along2");
       boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
       boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
       ptr_t func                     = concatenate(inp0, inp1, 2);

       determine_shapes(*func);
       BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
       BOOST_CHECK_EQUAL(func->result()->shape.at(0), 4);
       BOOST_CHECK_EQUAL(func->result()->shape.at(1),5);
       BOOST_CHECK_EQUAL(func->result()->shape.at(2),16);

       derivative_tester(*func).epsilon(epsilon).test();
    }
    {
       TRACE(g_log, "dim3along0b");
      boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[6][5][8]);
      boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[4][5][8]);
      ptr_t func                     = concatenate(inp0, inp1, 0);

      determine_shapes(*func);
      BOOST_CHECK_EQUAL(func->result()->shape.size(), 3);
      BOOST_CHECK_EQUAL(func->result()->shape.at(0), 10);
      BOOST_CHECK_EQUAL(func->result()->shape.at(1),5);
      BOOST_CHECK_EQUAL(func->result()->shape.at(2),8);

      derivative_tester(*func).epsilon(epsilon).test();
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
   derivative_testing::derivative_tester(*pool).test();
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
            //typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
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
            derivative_testing::derivative_tester(*op).verbose().test();
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
                                BOOST_CHECK_SMALL(fabs((a/(float)x) - b) , 0.0001);  
                            }
                        }
                    }
            derivative_tester(*op).verbose().test();
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
            //typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
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
            derivative_tester(*op).verbose().test();
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
            //typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
            typedef boost::shared_ptr<Op> op_ptr;
            
            std::vector< op_ptr >  input(n);       
            boost::shared_ptr<ParameterInput> in1;
            boost::shared_ptr<ParameterInput> in2;
            boost::shared_ptr<ParameterInput> in3;
            
        //generate all inputs and fill them with rand vals
        for ( unsigned int i = 0; i < n; i++){
                boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[x][y][z], "x-"+boost::lexical_cast<std::string>(i));
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
            derivative_tester(*op).epsilon(1.).verbose().test();
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
            //typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
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
        derivative_tester(*op).epsilon(1.).verbose().test();
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
            //typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
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
            derivative_tester(*op).epsilon(1.).verbose().test();
cuv::safeThreadSync();
}

BOOST_AUTO_TEST_CASE( Concatenate_N_center_dim )
{  
            unsigned int n = 3;
            unsigned int x = 2;
            unsigned int y = 3;
            unsigned int z = 2;
            unsigned int z1 = 5;

            using namespace cuv;
            using namespace cuvnet;
            //typedef boost::shared_ptr<cuvnet::Concatenate> ptr_t;
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
            
            op_ptr op = concatenate(input, 1);
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

        std::vector<unsigned int> desired_shape = {x, n*y, z, z1};
        cuvAssert(out_op->cdata().shape() == desired_shape);        
        
        for ( unsigned int j = 0; j < x; j++)
            for ( unsigned int k = 0; k < y; k++)
                for ( unsigned int h = 0; h < z; h++)
                    for (unsigned int l = 0; l < z1; l++)

                        for ( unsigned int i = 0; i < n; i++){
                            float  a = boost::dynamic_pointer_cast<ParameterInput>(input[i])->data()(j,k,h,l);
                            float  b = out_op->cdata()(j,i*n+k,h,l);
                            BOOST_CHECK_SMALL(fabs(a - b) , 0.0001);  
                        }
                            
        derivative_tester(*op).epsilon(1.).verbose().test();
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
   derivative_tester(*op).test();
   cuv::safeThreadSync();    
}

BOOST_AUTO_TEST_CASE(classloss) {
	typedef boost::shared_ptr<ParameterInput> inp_ptr_t;
	typedef boost::shared_ptr<Op> op_ptr_t;
   
    for (int i = 0; i < 4; i++) // iterate over 4 subcases
    {
        bool use_ignore = i%2;
        bool first = i < 2;
        LOG4CXX_WARN(g_log, "2 dimensions, first: " << first << ", ignore: " << use_ignore  );

        int n_class = 3;
        int n_batch = 5;
        inp_ptr_t out, tch, ign;
        if (first) {
            out = boost::make_shared<ParameterInput>(cuv::extents[n_class][n_batch]);
            tch = boost::make_shared<ParameterInput>(cuv::extents[n_class][n_batch]);
            ign = boost::make_shared<ParameterInput>(cuv::extents[1      ][n_batch]);
        } else {
            out = boost::make_shared<ParameterInput>(cuv::extents[n_batch][n_class]);
            tch = boost::make_shared<ParameterInput>(cuv::extents[n_batch][n_class]);
            ign = boost::make_shared<ParameterInput>(cuv::extents[n_batch][1      ]);
        }

        out->set_derivable(false);
        tch->set_derivable(false);
        ign->set_derivable(false);

        op_ptr_t loss;
        if (use_ignore)
            loss = boost::make_shared<ClassificationLoss>(out->result(), tch->result(), ign->result(), first ? 0 : 1);
        else
            loss = boost::make_shared<ClassificationLoss>(out->result(), tch->result(), first ? 0 : 1);

        out->data() = 0.f;
        tch->data() = 0.f;
        double tot = 0;
        double pos = 0;

        for (int b = 0; b < n_batch; b++) {
            int t = rand() % n_class;
            int o = rand() % n_class;
            float i = drand48();

            if (first) {
                out->data()(o, b) = 1.0f;
                tch->data()(t, b) = 1.0f;
                ign->data()(0, b) = (float) i;
            } else {
                out->data()(b, o) = 1.0f;
                tch->data()(b, t) = 1.0f;
                ign->data()(b, 0) = (float) i;
            }

            if (use_ignore) {
                tot += i;
                pos += i * (t == o);
            } else {
                tot += 1;
                pos += t == o;
            }
        }

        function func(loss, 0);
        func.evaluate();
        
        BOOST_CHECK_CLOSE((float) func.result()(0), 1.f - ((float) pos / tot), 0.001 );
        derivative_tester(*loss).test();
    }
    
    for (int i = 0; i < 4; i++) // iterate over 4 subcases
    {
        bool use_ignore = i%2;
        bool first = i < 2;
        LOG4CXX_WARN(g_log, "4 dimensions, first: " << first << ", ignore: " << use_ignore  );

        int n_class = 3;
        int n_pixel = 4;
        int n_batch = 5;
        inp_ptr_t out, tch, ign;
        if (first) {
            out = boost::make_shared<ParameterInput>(cuv::extents[n_class][n_pixel][n_pixel][n_batch]);
            tch = boost::make_shared<ParameterInput>(cuv::extents[n_class][n_pixel][n_pixel][n_batch]);
            ign = boost::make_shared<ParameterInput>(cuv::extents[1      ][n_pixel][n_pixel][n_batch]);
        } else {
            out = boost::make_shared<ParameterInput>(cuv::extents[n_batch][n_pixel][n_pixel][n_class]);
            tch = boost::make_shared<ParameterInput>(cuv::extents[n_batch][n_pixel][n_pixel][n_class]);
            ign = boost::make_shared<ParameterInput>(cuv::extents[n_batch][n_pixel][n_pixel][1      ]);
        }

        out->set_derivable(false);
        tch->set_derivable(false);
        ign->set_derivable(false);

        op_ptr_t loss;
        if (use_ignore)
            loss = boost::make_shared<ClassificationLoss>(out->result(), tch->result(), ign->result(), first ? 0 : 3);
        else
            loss = boost::make_shared<ClassificationLoss>(out->result(), tch->result(), first ? 0 : 3);

        out->data() = 0.f;
        tch->data() = 0.f;
        double tot = 0;
        double pos = 0;

        for (int b = 0; b < n_batch; b++) {
            for (int p1 = 0; p1 < n_pixel; p1++) {
                for (int p2 = 0; p2 < n_pixel; p2++) {
                    int t = rand() % n_class;
                    int o = rand() % n_class;
                    float i = drand48();

                    if (first) {
                        out->data()(o, p1, p2, b) = 1.0f;
                        tch->data()(t, p1, p2, b) = 1.0f;
                        ign->data()(0, p1, p2, b) = (float) i;
                    } else {
                        out->data()(b, p1, p2, o) = 1.0f;
                        tch->data()(b, p1, p2, t) = 1.0f;
                        ign->data()(b, p1, p2, 0) = (float) i;
                    }

                    if (use_ignore) {
                        tot += i;
                        pos += i * (t == o);
                    } else {
                        tot += 1;
                        pos += t == o;
                    }
                }
            }
        }

        function func(loss, 0);
        func.evaluate();
        
        BOOST_CHECK_CLOSE((float) func.result()(0), 1.f - ((float) pos / tot), 0.001 );
        derivative_tester(*loss).test();
    }
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
                    derivative_tester(*func).test();
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
                 derivative_tester(*func).test();
               }
               {
                  ptr_t func                       = boost::make_shared<Convolve2dTheano>(inp0->result(), inp1->result(), "full");
                  derivative_tester(*func).test();
               }
               {
                 ptr_t func                       = boost::make_shared<Convolve2dTheano>(inp0->result(), inp1->result(), padding_bias->result(), "full");
                 derivative_tester(*func).test();
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
                    derivative_tester(*func).test();
                }
                {
                    ptr_t func                       = boost::make_shared<ShuffleDim>(inp0->result(), cuv::extents[0][1][3][2]);
                    derivative_tester(*func).test();
                }
            }
        }
}

BOOST_AUTO_TEST_CASE(upscale){
    typedef boost::shared_ptr<Op> ptr_t;
    using namespace cuv;
    using namespace cuv::misc_conv;
        {
            unsigned int nImgChan = 4;     
            unsigned int nImgPixX = 5;
            unsigned int nImgPixY = 5;
            unsigned int nImg     = 3;
            boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixX][nImgPixY][nImg] , "inputs");
            ptr_t func = boost::make_shared<Upscale>(inp0->result(), 4);
            derivative_tester(*func).verbose().test();
        }
}


#endif

BOOST_AUTO_TEST_CASE(cuDNN_convolve){
    typedef boost::shared_ptr<Op> ptr_t;

        {
            unsigned int nImgChan = 3;
            unsigned int nImgPixX = 9;
            unsigned int nImgPixY = 9;
            unsigned int nImg     = 2;

            unsigned int nFiltChan = nImgChan;
            unsigned int nFiltPixX  = 3;
            unsigned int nFiltPixY  = 3;
            unsigned int nFilt     = 1;

            unsigned int padding_x = 1;
            unsigned int padding_y = 1;

            unsigned int hor_stride = 2;
            unsigned int ver_stride = 2;

            {
               boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX], "inputs");
               boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFilt][nFiltChan][nFiltPixY][nFiltPixX], "weights");

          /*     {
                 ptr_t func                       = boost::make_shared<ConvolvecuDNN>(inp0->result(), inp1->result());
                 derivative_tester(*func).test();
               }*/
               {
                  ptr_t func                       = boost::make_shared<ConvolvecuDNN>(inp0->result(), inp1->result(), padding_y, padding_x, ver_stride, hor_stride);
                  derivative_tester(*func).epsilon(1.).full_jacobian().no_state_precision(1e-6).test();
                }
            }
        }
}

BOOST_AUTO_TEST_CASE(cuDNN_speed){
    typedef boost::shared_ptr<Op> ptr_t;

        {
            unsigned int nImgChan = 3;
            unsigned int nImgPixX = 224;
            unsigned int nImgPixY = 224;
            unsigned int nImg     = 64;

            unsigned int nFiltChan = nImgChan;
            unsigned int nFiltPixX  = 11;
            unsigned int nFiltPixY  = 11;
            unsigned int nFilt     = 32;

            {
               boost::shared_ptr<ParameterInput>  inp0a = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX], "inputs");
               boost::shared_ptr<ParameterInput>  inp0b = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixY][nImgPixX][nImg], "inputs");
               boost::shared_ptr<ParameterInput>  inp1a = boost::make_shared<ParameterInput>(cuv::extents[nFilt][nFiltChan][nFiltPixY][nFiltPixX], "weights");
               boost::shared_ptr<ParameterInput>  inp1b = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixY*nFiltPixX][nFilt], "weights");

               for(unsigned int dinput = 0; dinput < 2; dinput ++){
            	   std::cout << "derivative w.r.t. param: " << dinput << std::endl;
            	   int padding = 0;

            	   ptr_t op0 = boost::make_shared<ConvolvecuDNN>(inp0a->result(), inp1a->result(),padding, padding);
            	   ptr_t op1 = boost::make_shared<Convolve>(inp0b->result(), inp1b->result(), true, padding, 1, 1, 0);

            	   cuvnet::function func0f(op0);
            	   MEASURE_TIME(cudnn_f, func0f.evaluate(), 10);
            	   cuvnet::function func1f(op1);
            	   MEASURE_TIME(alex_f, func1f.evaluate(), 10);

            	   cuvnet::delta_function func0(op0, op0, 0, dinput);
            	   MEASURE_TIME(cudnn_fb, func0.evaluate(), 10);

            	   cuvnet::delta_function func1(op1, op1, 0, dinput);
            	   MEASURE_TIME(alex_fb, func1.evaluate(), 10);

            	   std::cout << "fprop speedup alex/cudnn: " << (alex_f) / (cudnn_f) << std::endl;
            	   std::cout << "bprop speedup alex/cudnn: " << (alex_fb-alex_f) / (cudnn_fb-cudnn_f) << std::endl;
               }
            }
        }
}

BOOST_AUTO_TEST_CASE(cuDNN_streams_speed)
{
	using namespace std;

    typedef boost::shared_ptr<Op> ptr_t;

    unsigned int nImgChan = 3;
    unsigned int nImgPixX = 1000;
    unsigned int nImgPixY = 1000;
    unsigned int nImg     = 5;

    unsigned int nFiltChan = nImgChan;
    unsigned int nFiltPixX  = 10;
    unsigned int nFiltPixY  = 10;
    unsigned int nFilt     = 1;

    unsigned int bias1 = 1;
    unsigned int bias2 = nFilt;
    unsigned int bias3 = 1;
    unsigned int bias4 = 1;

    boost::shared_ptr<ParameterInput>  in1 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX], "inputs");
    boost::shared_ptr<ParameterInput>  in2 = boost::make_shared<ParameterInput>(cuv::extents[nFilt][nFiltChan][nFiltPixY][nFiltPixX], "weights");
    boost::shared_ptr<ParameterInput>  in3 = boost::make_shared<ParameterInput>(cuv::extents[bias1][bias2][bias3][bias4], "bias");

    fill_rnd_uniform(in1->data());
    fill_rnd_uniform(in2->data());
    fill_rnd_uniform(in3->data());

	int padding = 0;
 	ptr_t op = boost::make_shared<ConvolvecuDNN>(in1->result(), in2->result(),in3->result(),padding, padding);

    // assumption: op has only one result
	boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op->result());

	// tell that we want derivative w.r.t. all params

	std::vector<Op*> params(3);
	params[0] = in1.get();
	params[1] = in2.get();
	params[2] = in3.get();

	swiper swipe(*op, 0, params);

	swipe.fprop();
	cuvAssert(!cuv::has_nan(out_op->cdata()));
	cuvAssert(!cuv::has_inf(out_op->cdata()));

	MEASURE_TIME(cudnn_b, swipe.bprop(), 10);
	std::cout << "bprop cudnn: " << cudnn_b << std::endl;

	//speeds one vs multiple streams: 419494 vs 343819 -> speedup: 1.22

}


BOOST_AUTO_TEST_CASE(cuDNN_pooling){
    typedef boost::shared_ptr<Op> ptr_t;

    unsigned int nImgChan = 1;
    unsigned int nImgPixX  = 6;
    unsigned int nImgPixY  = 6;
    unsigned int nImg     = 1;

    unsigned int window_height = 4;
    unsigned int window_width = 4;

    unsigned int vertical_stride = 2;
    unsigned int horizontal_stride = 2;

    unsigned int vertical_padding = 0;
    unsigned int horizontal_padding = 0;

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImg][nImgChan][nImgPixY][nImgPixX]);
        ptr_t func = boost::make_shared<PoolingcuDNN>(inp0->result(), cuv::alex_conv::PT_MAX, window_height, window_width, vertical_stride, horizontal_stride, vertical_padding, horizontal_padding);
/*
	double s = inp0->data().size();
	double epsilon = 0.01;
    derivative_tester(*func).values(-s/2,s/2).spread_values().epsilon(epsilon).full_jacobian().only_variant(4).test();
*/
	 derivative_tester(*func).test();

    }
}


BOOST_AUTO_TEST_CASE(szegedy_op){
    //typedef boost::shared_ptr<Op> ptr_t;

    unsigned int bs =  10; // batch size
    unsigned int  K =   4; // number of predictions per image
    unsigned int  C =   2; // number of classes
    unsigned int  T =   2; // teacher bboxes
    float     alpha = 1.0; // scales bounding box distance loss

    boost::shared_ptr<ParameterInput> inp0 = boost::make_shared<ParameterInput>(cuv::extents[bs][C*K*5]);

    std::vector<std::vector<datasets::rotated_rect> > kmeans(K);
    for (unsigned int c = 0; c < C; c++) {
        kmeans[c].resize(K);
        for (unsigned int k = 0; k < K; k++) {
            kmeans[c][k].x = drand48() * 1;
            kmeans[c][k].y = drand48() * 1;
            kmeans[c][k].h = drand48() * 1;
            kmeans[c][k].w = drand48() * 1;
        }
    }
    std::vector<std::vector<datasets::bbox> > teach(bs);
    for (unsigned int b = 0; b < bs; b++) {
        teach[b].resize(T);
        for (unsigned int t = 0; t < T; t++) {
            teach[b][t].rect.x = drand48();
            teach[b][t].rect.y = drand48();
            teach[b][t].rect.h = drand48();
            teach[b][t].rect.w = drand48();
            teach[b][t].klass = drand48() * C;
        }
    }

    cuv::fill_rnd_uniform(inp0->data());
    inp0->data() *= 2.f;
    inp0->data() -= 1.f;

    // first teacher is exactly the mean
    teach[0][0].rect = kmeans[0][0];
    // accordingly, the input should be very confident and have zero offsets
    inp0->data()(0, 0) = 0.f;
    inp0->data()(0, 1) = 0.f;
    inp0->data()(0, 2) = 0.f;
    inp0->data()(0, 3) = 0.f;
    inp0->data()(0, 4) = 5.f;

    boost::shared_ptr<BoundingBoxMatching> func = 
        boost::make_shared<BoundingBoxMatching>(inp0->result(), kmeans, alpha, C); 
    
    // set teacher
    //func->set_teacher_bbox(teach);
    func->m_teach = teach;

    function f(func);
    matrix res = f.evaluate();
   
    BOOST_CHECK_EQUAL(res[0], func->get_f_match() / bs + func->get_f_conf() / bs / alpha);

    std::vector<std::vector<unsigned int> > n_matched;
    n_matched.resize(bs);
    for (unsigned int b = 0; b < bs; b++) {
        n_matched[b].resize(T, 0);
        for (unsigned int c = 0; c < C; c++) {
        for (unsigned int k = 0; k < K; k++) {
            int i_m = func->m_matching[b][K*c + k]; 
            if(i_m >= 0) {
                n_matched[b][i_m]++;
                BOOST_CHECK_EQUAL(teach[b][i_m].klass, c); // teacher matched wrt to class
            }
        }
        }
    }

    for(auto mv : n_matched)
       for (auto m : mv)
           BOOST_CHECK_EQUAL(m, 1); // each output matched once

    //std::vector<std::vector<datasets::bbox> > output_bbox = func->get_output_bbox(); 
    //for (unsigned int b = 0; b < bs; b++) {
    //    for (unsigned int k = 0; k < K; k++) {
    //        BOOST_CHECK_EQUAL(output_bbox[b][k].rect.x, inp0->data()(b, k * 5 + 0) + kmeans[k].x);
    //        BOOST_CHECK_EQUAL(output_bbox[b][k].rect.y, inp0->data()(b, k * 5 + 1) + kmeans[k].y);
    //        BOOST_CHECK_EQUAL(output_bbox[b][k].rect.h, inp0->data()(b, k * 5 + 2) + kmeans[k].h);
    //        BOOST_CHECK_EQUAL(output_bbox[b][k].rect.w, inp0->data()(b, k * 5 + 3) + kmeans[k].w);
    //    }
    //}
    //std::cout << "loss: " << func->get_f_match() << " " << func->get_f_conf() << std::endl;
   
    derivative_tester (*func).values(1,-1).test();
}

BOOST_AUTO_TEST_SUITE_END()




