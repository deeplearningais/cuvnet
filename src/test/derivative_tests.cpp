#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>

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
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Sum>(inp0->result());
    derivative_tester(*func,false,0.015);
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
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.3, -2.5);
    func                           = boost::make_shared<Pow>(2.f,func->result());
    derivative_tester(*func);
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
