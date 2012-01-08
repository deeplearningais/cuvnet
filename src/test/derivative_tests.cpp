#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>

#include <cuvnet/ops/axpby.hpp>
#include <cuvnet/ops/identity.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/mat_plus_vec.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/pow.hpp>
#include <cuvnet/ops/prod.hpp>
#include <cuvnet/ops/tanh.hpp>

using namespace cuvnet;
using std::printf;

#define PM(X) print(#X,X);

void print(const std::string& s, const matrix& M){
    std::cout << "_________________________________________"<<std::endl;
    std::cout << "------------ "<<s<<" (";
    std::copy(M.shape().begin(), M.shape().end(),std::ostream_iterator<unsigned int>(std::cout,","));
    std::cout << ") ------------"<<std::endl;
    if(M.ndim()==1){
        unsigned int cnt=0;
        for (unsigned int i = 0; i < M.size(); ++i){
            printf("% 2.2f ", (float)M[i]);
            if((cnt++*6)>80){printf("\n");cnt=0;}
        }
    }
    if(M.ndim()==2){
        for (unsigned int i = 0; i < M.shape(0); ++i){
            printf("   ");
            unsigned int cnt=0;
            for (unsigned int j = 0; j < M.shape(1); ++j){
                printf("% 2.2f ", (float)M(i,j));
                if((cnt++*6)>80){printf("\n");cnt=0;}
            }
            printf("\n");
        }
    }
}

void set_delta_to_unit_vec(Op::result_t& r, unsigned int i){
    r->delta.reset(new matrix(r->shape));
    r->delta.data()    = 0.f;
    r->delta.data()[i] = 1.f;
}
unsigned int prod(const std::vector<unsigned int>& v){
    return std::accumulate(v.begin(),v.end(),1u, std::multiplies<unsigned int>());
}
void derivative_tester(Op& op){
    // assumption: op has only one result
    boost::shared_ptr<Output> out_op = boost::make_shared<Output>(op.result());

    // tell that we want derivative w.r.t. all params
    param_collector_visitor pcv;
    op.visit(pcv);

    // fill all params with random numbers
    BOOST_FOREACH(Op* raw, pcv.plist){
        Input* param = dynamic_cast<Input*>(raw);
        EXPECT_TRUE(param!=NULL);
        for (int i = 0; i < param->data().size(); ++i)
        {
            //param->data()[i] = 2.f;
            param->data()[i] = (float)drand48();
        }
    }

    swiper swipe(op, true, pcv.plist);

    BOOST_FOREACH(Op* raw, pcv.plist){
        Input* param = dynamic_cast<Input*>(raw);
	EXPECT_TRUE(param!=NULL);
        unsigned int n_inputs  = param->data().size();
        unsigned int n_outputs = prod(op.result()->shape);
        matrix J(n_outputs, n_inputs); J = 0.f;
        for(unsigned int out=0;out<n_outputs;out++){
            swipe.fprop();
            set_delta_to_unit_vec(op.result(),out);
            swipe.bprop();

            // set row in J to the backpropagated value
            matrix d_in = param->result()->delta.cdata();
            d_in.reshape(cuv::extents[n_inputs]);
            matrix Jrow(cuv::indices[cuv::index_range(out,out+1)][cuv::index_range()], J);
            Jrow = d_in;
        }

        matrix J_(n_inputs,n_outputs); J_ = 0.f;
        for (unsigned int in = 0; in < n_inputs; ++in) {
            static const float eps = 0.0001f;
            float v = param->data()[in];
            param->data()[in] = v + eps;
            swipe.fprop();
            matrix o_plus     = out_op->cdata(); 
            param->data()[in] = v - eps;
            swipe.fprop();
            matrix o_minus    = out_op->cdata();
            param->data()[in] = v;

            o_plus .reshape(cuv::extents[n_outputs]);
            o_minus.reshape(cuv::extents[n_outputs]);
            o_plus -= o_minus;
            o_plus /= 2.f*eps;

            // set row in J_ to finite-difference approximation
            matrix J_row(cuv::indices[cuv::index_range(in,in+1)][cuv::index_range()], J_);
            J_row = o_plus;
        }
        matrix J_t(n_outputs, n_inputs);
	cuv::transpose(J_t,J_);
        //PM(J_); PM(J);
        EXPECT_NEAR(cuv::norm2(J_t-J), 0.f, 0.01f );
    }
}

TEST(derivative_test, derivative_test_pow){
   typedef boost::shared_ptr<Op> ptr_t;
   boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
   ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());
   derivative_tester(*pow);
}

TEST(derivative_test, derivative_test_tanh){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Tanh>(inp->result());
    derivative_tester(*func);
}

TEST(derivative_test, derivative_test_axpby){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.3, -2.5);
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

	    derivative_tester(*func);
    }
    {
	    boost::shared_ptr<Input>  inp0 = boost::make_shared<Input>(cuv::extents[3][5]);
	    boost::shared_ptr<Input>  inp1 = boost::make_shared<Input>(cuv::extents[5]);
	    ptr_t func		           = boost::make_shared<MatPlusVec>(inp0->result(), inp1->result(), true);

	    derivative_tester(*func);
    }
}
