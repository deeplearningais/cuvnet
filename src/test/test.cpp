#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include <cuvnet/op.hpp>

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
    Op::param_collector_visitor pcv;
    op.visit(pcv);

    // fill all params with random numbers
    BOOST_FOREACH(Op* raw, pcv.plist){
        Input* param = dynamic_cast<Input*>(raw);
        EXPECT_TRUE(param!=NULL);
        for (int i = 0; i < param->data().size(); ++i)
        {
            param->data()[i] = 2.f;//(float)drand48();
        }
    }

    Op::swiper swipe(op, true, pcv.plist);

    BOOST_FOREACH(Op* raw, pcv.plist){
        Input* param = dynamic_cast<Input*>(raw);
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
        matrix J_t(n_inputs, n_outputs);
        //PM(J_); PM(J);
        EXPECT_NEAR(cuv::norm2(J_-J), 0.f, 0.0001f );
    }
}

TEST(ce_ptr_test, init){
	ce_ptr<int> k;
	assert(!k);

	ce_ptr<int> i = ce_ptr<int>(new int(4));
	ce_ptr<int> j = i;
	
	EXPECT_TRUE(i);
	EXPECT_TRUE(j);
	EXPECT_EQ(i.ptr() , j.ptr());

	void* p = (void*) 4;
	void* q = (void*) 5;
	i.lock(p); // p says he wants to keep the value of i

	EXPECT_TRUE(i.writable(p));
	EXPECT_TRUE(j.writable(p));

	EXPECT_TRUE(!i.writable(q));
	EXPECT_TRUE(!j.writable(q));

	EXPECT_TRUE(i.locked_by(p));
	EXPECT_TRUE(j.locked_by(p));
	EXPECT_TRUE(!i.locked_by(q));
	EXPECT_TRUE(!j.locked_by(q));

	std::cout<<i << " "<< i.ptr()<<std::endl;
	std::cout<<j << " "<< j.ptr()<<std::endl;

	j.lock(q);     // q says he wants to keep the value of j

	EXPECT_TRUE(i.locked_by(q)); // both are locked by q now
	EXPECT_TRUE(j.locked_by(q)); // both are locked by q now

	EXPECT_TRUE(i.locked_by(p)); // both are locked by p still
	EXPECT_TRUE(j.locked_by(p)); // both are locked by p still

	EXPECT_TRUE(!j.writable(p)); // noone can write w/o copying
	EXPECT_TRUE(!j.writable(q)); // noone can write w/o copying

	j.data(q) = 5;          // q wants to change the value of j.
	                        // --> j gets detached
				// --> i is not locked by q anymore
				// --> j is not locked by p anymore
	j.reset(new int(6));
	j.lock(q);

	EXPECT_TRUE(!i.locked_by(q)); 
	EXPECT_TRUE(!j.locked_by(p));
	EXPECT_TRUE(i.locked_by(p)); 
	EXPECT_TRUE(j.locked_by(q));

	EXPECT_TRUE(i.writable(p));
	EXPECT_TRUE(j.writable(q));

	EXPECT_TRUE(!i.writable(q));
	EXPECT_TRUE(!j.writable(p));

	EXPECT_NE(i.ptr() , j.ptr());
	EXPECT_EQ(*i , 4);
	EXPECT_EQ(*j , 6);

	std::cout<<i << " "<< i.ptr()<<std::endl;
	std::cout<<j << " "<< j.ptr()<<std::endl;

}

TEST(op_test,wiring){
	typedef boost::shared_ptr<Op> ptr_t;
    ptr_t inp = boost::make_shared<Input>(cuv::extents[10][20]);
    ptr_t id  = boost::make_shared<Identity>(inp->result());

    // find input object
    Op::param_collector_visitor pcv;
    id->visit(pcv);
    EXPECT_EQ(pcv.plist.size(),1);

    // determine which derivatives to calculate
    EXPECT_EQ(id->param(0)->need_derivative, false);
    id->set_calculate_derivative(pcv.plist);
    EXPECT_EQ(id->param(0)->need_derivative, true);

    // shape propagation
    id->visit(Op::determine_shapes_visitor());
    EXPECT_EQ(*pcv.plist.begin(),inp.get());
    EXPECT_EQ(id->result(0)->shape.at(0), 10);
    EXPECT_EQ(id->result(0)->shape.at(1), 20);
}

TEST(op_test,fprop_and_bprop){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[10][20]);
    ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());
    boost::shared_ptr<Output> out = boost::make_shared<Output>(pow->result());

    boost::dynamic_pointer_cast<Input>(inp)->data() = 2.f;

    // tell that we want derivative w.r.t. all params
    Op::param_collector_visitor pcv;
    pow->visit(pcv);
    EXPECT_EQ(pcv.plist.size(),1);
    pow->set_calculate_derivative(pcv.plist);
    EXPECT_EQ(pow->param(0)->need_derivative, true);
    pow->visit(Op::determine_shapes_visitor());
    EXPECT_EQ(pow->result()->shape[0],10);
    EXPECT_EQ(pow->result()->shape[1],20);

    inp->fprop();
    pow->fprop();
    out->fprop();
    EXPECT_EQ(out->cdata()[0],4);

    set_delta_to_unit_vec(pow->result(), 0);
    pow->bprop();
    inp->bprop();
    EXPECT_EQ(inp->result()->delta.cdata()[0],4);
}

TEST(op_test,toposort){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[10][20]);
    ptr_t id                      = boost::make_shared<Identity>(inp->result());
    ptr_t pow                     = boost::make_shared<Pow>(2,id->result());
    boost::shared_ptr<Output> out = boost::make_shared<Output>(pow->result());

    // tell that we want derivative w.r.t. all params
    Op::param_collector_visitor pcv;
    pow->visit(pcv);
    EXPECT_EQ(pcv.plist.size(),1);
    pow->set_calculate_derivative(pcv.plist);

    // determine sequence in which to call fwd, bwd pass
    Op::toposort_visitor tv(true);
    pow->visit(tv);

    EXPECT_EQ(tv.plist.size(), 3);
    EXPECT_EQ(tv.plist.at(0), inp.get());
    EXPECT_EQ(tv.plist.at(1), id.get());
    EXPECT_EQ(tv.plist.at(2), pow.get());
}

//TEST(op_test, derivative_test_pow){
//    typedef boost::shared_ptr<Op> ptr_t;
//    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
//    ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());
//    derivative_tester(*pow);
//}

TEST(op_test, derivative_test_tanh){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[3][5]);
    ptr_t func                    = boost::make_shared<Tanh>(inp->result());
    derivative_tester(*func);
}
