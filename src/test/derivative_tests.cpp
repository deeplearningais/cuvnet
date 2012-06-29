#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <boost/assign.hpp>
#include <gtest/gtest.h>

#define CUVNET_PRECISE_SUM 1

#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>

#include <cuvnet/ops.hpp>

using namespace cuvnet;
using std::printf;
using namespace cuvnet::derivative_testing;

TEST(op_test, deltasink){
    typedef boost::shared_ptr<Op> ptr_t;
    typedef boost::shared_ptr<ParameterInput> param_t;
    param_t inp = boost::make_shared<ParameterInput>(cuv::extents[2][4]);
    inp->data() = 3.f;
    ptr_t func  = boost::make_shared<Pow>(2.f,inp->result());
    boost::shared_ptr<DeltaSink> ds = delta_sink("pow_delta", inp); // monitor the delta 

    swiper s(*func, 0, boost::assign::list_of<Op*>(inp.get()));
    s.fprop();
    s.bprop();
    EXPECT_NEAR(ds->cdata()[0], 6.f, 0.001f);
    s.fprop();
    s.bprop();
    EXPECT_NEAR(ds->cdata()[0], 6.f, 0.001f); // make sure it does not add to previous value
}

TEST(derivative_test, derivative_test_pipe){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Pipe>(inp->result(),0);
   derivative_tester(*func);
}
TEST(derivative_test, derivative_test_row_select){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);

   ptr_t func                    = row_select(inp0,inp1,1);

   derivative_tester(*result(func,0));
   derivative_tester(*result(func,1));
}
TEST(derivative_test, derivative_test_pow){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());
   derivative_tester(*pow);
}
TEST(derivative_test, derivative_test_exp){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t exp                     = boost::make_shared<Exp>(2,inp->result());
   derivative_tester(*exp,0,false,0.01);
}
TEST(derivative_test, derivative_test_abs){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Abs>(inp->result());
   derivative_tester(*func);
}

TEST(derivative_test, derivative_test_log){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
   boost::shared_ptr<Pow>    pw  = boost::make_shared<Pow>(2, inp->result());
   ptr_t func                    = boost::make_shared<Log>(pw->result());
   derivative_tester(*func);
}

TEST(derivative_test, derivative_test_mean){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Mean>(inp->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_tanh){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Tanh>(inp->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_sin){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Sin>(inp->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_cos){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Cos>(inp->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_add_scalar){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<AddScalar>(1.f,inp->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_mult_scalar){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<MultScalar>(1.5f,inp->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_subtract_from_scalar){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    //ptr_t func                    = boost::make_shared<SubtractFromScalar>(1.f,inp->result());
    ptr_t func                    = 1.f-inp;
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_multiply){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Multiply>(inp0->result(), inp1->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_atan2){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5], "y");
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5], "x");
    ptr_t func                     = boost::make_shared<Atan2>(inp0->result(), inp1->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_eps_insensitive_loss){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[20]);
    inp0->set_derivable(false);
    ptr_t func                     = boost::make_shared<EpsilonInsensitiveLoss>(0.1, inp0->result(), inp1->result());
    derivative_tester(*func, 0, true, .003, 0.0, 1.0);
}
TEST(derivative_test, derivative_test_neg_cross_entropy_logistic){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[5]);
    ptr_t func                     = boost::make_shared<NegCrossEntropyOfLogistic>(inp0->result(), inp1->result());
    derivative_tester(*func);
}
TEST(derivative_test, bernoulli_kl){
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
TEST(derivative_test, derivative_test_axpby){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.3, -2.5);
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_sum_mat_to_vec){
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
}
TEST(derivative_test, derivative_test_sum_mat_to_vec3d){
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
 *TEST(derivative_test, derivative_test_noiser){
 *        cuv::initialize_mersenne_twister_seeds();
 *        typedef boost::shared_ptr<Op> ptr_t;
 *    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[4][5]);
 *    ptr_t func                     = boost::make_shared<Noiser>(inp0->result());
 *    derivative_tester(*func);
 *}
 */
TEST(derivative_test, derivative_test_sum){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[2][2]);
    ptr_t func                     = boost::make_shared<Sum>(inp0->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_prod){
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

TEST(derivative_test, derivative_test_sq_axpby){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp0->result(), 1.5f, -1.8f);
    func                           = boost::make_shared<Pow>(2.f,func->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_xtx){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = inp0*inp0;
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_xt1mx){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = inp0*(1.f-inp0);
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_add_to_param){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t tmp0                     = boost::make_shared<Pow>(2.f, inp0->result());
    ptr_t tmp1                     = boost::make_shared<Pow>(3.f, inp0->result());
    ptr_t func                     = boost::make_shared<Sum>(tmp0->result());
    add_to_param(func, tmp1);  // sum(  x^2+x^3 )
    derivative_tester(*func, 0, false, 0.03f);
}

TEST(derivative_test, derivative_test_softmax){
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

TEST(derivative_test, derivative_test_mll){
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
}

TEST(derivative_test, derivative_test_mat_plus_vec){
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
}
TEST(derivative_test, derivative_test_mat_times_vec){
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
}
TEST(derivative_test, derivative_test_convolve){
	typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;

    for (int padding = 0; padding < 2; ++padding)
    {
        {
            unsigned int nImgChan = 3;      // must be divisible by nGroups
            unsigned int nImgPixX = 16;
            unsigned int nImg     = 4;
            unsigned int nGroups  = 1;      // must be divisible by 2 ??

            unsigned int nFiltChan = nImgChan/nGroups;
            unsigned int nFiltPixX  = 3;
            unsigned int nFilt     = 16; 

            //unsigned int nResPix   = nImgPixX-nFiltPixX+1;

            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixX*nImgPixX][nImg], "inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt], "weights");
                ptr_t func                       = boost::make_shared<Convolve>(inp0->result(), inp1->result(), padding);

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
            unsigned int nImgPixX  = 16;
            unsigned int nImg     = 16;
            unsigned int nGroups  = 1;      // must be divisible by 2 ??

            unsigned int nFiltChan = nImgChan/nGroups;
            unsigned int nFiltPixX  = 3;
            unsigned int nFilt     = 1; 

            //unsigned int nResPix   = nImgPixX-nFiltPixX+1;
            {
                boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixX*nImgPixX][nImg],"inputs");
                boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt],"weights");
                ptr_t func                     = boost::make_shared<Convolve>(inp0->result(), inp1->result(), padding);

                derivative_tester(*func,0,false,.03);
            }
        }
    }
}

TEST(derivative_test, derivative_test_convolve_reorder){
	typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 1;      // must be divisible by nGroups
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??

    // we must set nGroups>1, so each filter will only be applied to nImgChan/nGroups inputs
    unsigned int nFiltChan = nImgChan/nGroups;
    unsigned int nFiltPixX  = 3;
    unsigned int nFilt     = 16; 

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixX*nImgPixX][nImg]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[nFiltChan][nFiltPixX*nFiltPixX][nFilt]);
    {
	    ptr_t func		               = boost::make_shared<ReorderForConv>(inp0->result());
	    derivative_tester(*func);
    }

    {
	    ptr_t func		               = boost::make_shared<ReorderFromConv>(inp1->result());
	    derivative_tester(*func);
    }
}

TEST(derivative_test, derivative_test_convolve_pooling){
	typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 16;      // must be multiple of 16 for bprop
    unsigned int nImgPixX  = 16;
    unsigned int nImg     = 1;

    {
	    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[nImgChan][nImgPixX*nImgPixX][nImg]);
	    ptr_t func		               = boost::make_shared<LocalPooling>(inp0->result(), cuv::alex_conv::PT_AVG);

	    derivative_tester(*func);
    }
}

TEST(derivative_test, derivative_test_flatten){
	typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result());

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 1);
        EXPECT_EQ(func->result()->shape.at(0), 8*3*3);

        derivative_tester(*func);
    }
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 2);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 2);
        EXPECT_EQ(func->result()->shape.at(0), 8);
        EXPECT_EQ(func->result()->shape.at(1), 3*3);

        derivative_tester(*func);
    }

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3][2]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 3);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 3);
        EXPECT_EQ(func->result()->shape.at(0), 8);
        EXPECT_EQ(func->result()->shape.at(1), 3);
        EXPECT_EQ(func->result()->shape.at(2), 3*2);

        derivative_tester(*func);
    }
}
TEST(derivative_test, derivative_test_reshape){
	typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3]);
        ptr_t func                     = boost::make_shared<Reshape>(inp0->result(), cuv::extents[3][8][3]);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 3);
        EXPECT_EQ(func->result()->shape.at(0), 3);
        EXPECT_EQ(func->result()->shape.at(1), 8);
        EXPECT_EQ(func->result()->shape.at(2), 3);

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

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 2);
        EXPECT_EQ(func->result()->shape.at(0), 8*3);
        EXPECT_EQ(func->result()->shape.at(1), 3);
    }

    {
        boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[8][3][3][2]);
        ptr_t func                       = boost::make_shared<Reshape>(inp0->result(), cuv::extents[8][-1][3]);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 3);
        EXPECT_EQ(func->result()->shape.at(0), 8);
        EXPECT_EQ(func->result()->shape.at(1), 3*2);
        EXPECT_EQ(func->result()->shape.at(2), 3);
    }
}
