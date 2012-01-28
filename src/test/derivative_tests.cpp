#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>

#define CUVNET_PRECISE_SUM 1

#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>

#include <cuvnet/ops.hpp>

using namespace cuvnet;
using std::printf;
using namespace cuvnet::derivative_testing;


TEST(derivative_test, derivative_test_pow){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
   ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());
   derivative_tester(*pow);
}

TEST(derivative_test, derivative_test_log){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
   ptr_t func                    = boost::make_shared<Log>(inp->result());
   derivative_tester(*func);
}

TEST(derivative_test, derivative_test_mean){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Mean>(inp->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_tanh){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Tanh>(inp->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_add_scalar){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<AddScalar>(1.f,inp->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_subtract_from_scalar){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
    //ptr_t func                    = boost::make_shared<SubtractFromScalar>(1.f,inp->result());
    ptr_t func                    = 1.f-inp;
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_multiply){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Multiply>(inp0->result(), inp1->result());
    derivative_tester(*func,false,0.025);
}
TEST(derivative_test, derivative_test_neg_cross_entropy_logistic){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[5]);
    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[5]);
    ptr_t func                     = boost::make_shared<NegCrossEntropyOfLogistic>(inp0->result(), inp1->result());
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_axpby){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.3, -2.5);
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_sum_mat_to_vec){
	typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),true);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<SumMatToVec>(inp0->result(),false);
        derivative_tester(*func,false,0.003);
    }
}
/*
 * // does not make much sense to test this
 *TEST(derivative_test, derivative_test_noiser){
 *        cuv::initialize_mersenne_twister_seeds();
 *        typedef boost::shared_ptr<Op> ptr_t;
 *    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[4][5]);
 *    ptr_t func                     = boost::make_shared<Noiser>(inp0->result());
 *    derivative_tester(*func);
 *}
 */
TEST(derivative_test, derivative_test_sum){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[2][2]);
    ptr_t func                     = boost::make_shared<Sum>(inp0->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_prod){
	typedef boost::shared_ptr<Op> ptr_t;
	{
		boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[5][3]);
		boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][8]);
		ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result());
		derivative_tester(*func);
	}
	{
		boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
		boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][8]);
		ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t');
		derivative_tester(*func);
	}
	{
		boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[5][3]);
		boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[8][3]);
		ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'n','t');
		derivative_tester(*func);
	}
	{
		boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
		boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[8][3]);
		ptr_t func		     = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
		derivative_tester(*func);
	}
	{
		boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
		boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[8][3]);
		ptr_t func0		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
		ptr_t func1		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
		ptr_t func 		       = boost::make_shared<Axpby>(func0->result(), func1->result(), 1.3,1.5);
		derivative_tester(*func);
	}
}

TEST(derivative_test, derivative_test_sq_axpby){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp0->result(), 1.5f, -1.8f);
    func                           = boost::make_shared<Pow>(2.f,func->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_xtx){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                     = inp0*inp0;
    derivative_tester(*func);
}
TEST(derivative_test, derivative_test_xt1mx){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                     = inp0*(1.f-inp0);
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_softmax){
	typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 0);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        ptr_t func                     = boost::make_shared<Softmax>(inp0->result(), 1);
        derivative_tester(*func);
    }
}

TEST(derivative_test, derivative_test_mll){
	typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func);
    }
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 1);
        derivative_tester(*func);
    }
    ///// SoftMax result of MultinomialLogisticLoss
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 0);
        derivative_tester(*func,1);
    }
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
        boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
        inp1->set_derivable(false);
        ptr_t func                     = boost::make_shared<MultinomialLogisticLoss>(inp0->result(), inp1->result(), 1);
        derivative_tester(*func,1);
    }
}

TEST(derivative_test, derivative_test_mat_plus_vec){
	typedef boost::shared_ptr<Op> ptr_t;
    {
	    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
	    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3]);
	    ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), false);

	    derivative_tester(*func,false,0.01f);
    }
    {
	    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
	    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[5]);
	    ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), true);

	    derivative_tester(*func,false,0.01f);
    }
}
TEST(derivative_test, derivative_test_convolve){
	typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 1;      // must be divisible by nGroups
    unsigned int nImgPix  = 16;
    unsigned int nImg     = 1;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??

    unsigned int nFiltChan = nImgChan/nGroups;
    unsigned int nFiltPix  = 3;
    unsigned int nFilt     = 16; 

    unsigned int nResPix   = nImgPix-nFiltPix+1;

    {
	    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);
	    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);
	    ptr_t func		           = boost::make_shared<Convolve>(inp0->result(), inp1->result());

	    derivative_tester(*func,false,0.01f);
    }
}

TEST(derivative_test, derivative_test_convolve_reorder){
	typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 1;      // must be divisible by nGroups
    unsigned int nImgPix  = 16;
    unsigned int nImg     = 1;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??

    // we must set nGroups>1, so each filter will only be applied to nImgChan/nGroups inputs
    unsigned int nFiltChan = nImgChan/nGroups;
    unsigned int nFiltPix  = 3;
    unsigned int nFilt     = 16; 

    {
	    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);
	    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);
	    ptr_t func		               = boost::make_shared<ReorderForConv>(inp0->result());

	    derivative_tester(*func,false,0.01f);
    }
}

TEST(derivative_test, derivative_test_convolve_pooling){
	typedef boost::shared_ptr<Op> ptr_t;

    using namespace cuv::alex_conv;
    unsigned int nImgChan = 16;      // must be multiple of 16 for bprop
    unsigned int nImgPix  = 16;
    unsigned int nImg     = 1;

    {
	    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);
	    ptr_t func		               = boost::make_shared<LocalPooling>(inp0->result(), cuv::alex_conv::PT_AVG);

	    derivative_tester(*func,false,0.01f);
    }
}

TEST(derivative_test, derivative_test_flatten){
	typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result());

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 1);
        EXPECT_EQ(func->result()->shape.at(0), 8*3*3);

        derivative_tester(*func,false,0.01f);
    }
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 2);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 2);
        EXPECT_EQ(func->result()->shape.at(0), 8);
        EXPECT_EQ(func->result()->shape.at(1), 3*3);

        derivative_tester(*func,false,0.01f);
    }

    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[8][3][3][2]);
        ptr_t func                       = boost::make_shared<Flatten>(inp0->result(), 3);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 3);
        EXPECT_EQ(func->result()->shape.at(0), 8);
        EXPECT_EQ(func->result()->shape.at(1), 3);
        EXPECT_EQ(func->result()->shape.at(2), 3*2);

        derivative_tester(*func,false,0.01f);
    }
}
TEST(derivative_test, derivative_test_reshape){
	typedef boost::shared_ptr<Op> ptr_t;
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[8][3][3]);
        ptr_t func                     = boost::make_shared<Reshape>(inp0->result(), cuv::extents[3][8][3]);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 3);
        EXPECT_EQ(func->result()->shape.at(0), 3);
        EXPECT_EQ(func->result()->shape.at(1), 8);
        EXPECT_EQ(func->result()->shape.at(2), 3);

        derivative_tester(*func,false,0.01f);
    }
    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[8][3][3]);
        ptr_t func                       = boost::make_shared<Reshape>(inp0->result(), cuv::extents[8*3][-1]);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 2);
        EXPECT_EQ(func->result()->shape.at(0), 8*3);
        EXPECT_EQ(func->result()->shape.at(1), 3);
    }

    {
        boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[8][3][3][2]);
        ptr_t func                       = boost::make_shared<Reshape>(inp0->result(), cuv::extents[8][-1][3]);

        func->visit(determine_shapes_visitor());
        EXPECT_EQ(func->result()->shape.size(), 3);
        EXPECT_EQ(func->result()->shape.at(0), 8);
        EXPECT_EQ(func->result()->shape.at(1), 3*2);
        EXPECT_EQ(func->result()->shape.at(2), 3);
    }
}
