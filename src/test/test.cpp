#include <stdexcept>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include <cuvnet/op.hpp>

using namespace cuvnet;

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

    inp->fprop();
    pow->fprop();
    out->fprop();
    EXPECT_EQ(out->data()[0],4);

    pow->bprop();
    inp->bprop();
    EXPECT_EQ(inp->result()->delta.cdata()[0],1);
}

TEST(op_test,toposort){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[10][20]);
    ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());
    boost::shared_ptr<Output> out = boost::make_shared<Output>(pow->result());

    // tell that we want derivative w.r.t. all params
    Op::param_collector_visitor pcv;
    pow->visit(pcv);
    EXPECT_EQ(pcv.plist.size(),1);
    pow->set_calculate_derivative(pcv.plist);

    // determine sequence in which to call fwd, bwd pass
    Op::toposort_visitor tv(true);
    pow->visit(tv);

    EXPECT_EQ(tv.plist.size(), 2);
    EXPECT_EQ(tv.plist.at(0), inp.get());
    EXPECT_EQ(tv.plist.at(1), pow.get());
}
