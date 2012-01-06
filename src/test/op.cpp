#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include <cuvnet/op.hpp>

using namespace cuvnet;
using std::printf;


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

    pow->result()->delta.reset(new matrix(pow->result()->shape));
    pow->result()->delta.data() = 1.f;
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


TEST(op_test, destruction){
	typedef boost::shared_ptr<Op> ptr_t;
	boost::shared_ptr<Input>  inp = boost::make_shared<Input>(cuv::extents[10][20]);
	ptr_t id                      = boost::make_shared<Identity>(inp->result());
	ptr_t pow                     = boost::make_shared<Pow>(2,id->result());

	boost::weak_ptr<Op> inp_cpy(inp);
	boost::weak_ptr<Op> id_cpy(id);
	boost::weak_ptr<Op> pow_cpy(pow);

	EXPECT_FALSE(!inp_cpy.lock());
	EXPECT_FALSE(!id_cpy.lock());
	EXPECT_FALSE(!pow_cpy.lock());

	inp.reset();
	id.reset();

	// hierarchy should /still/ exist!
	EXPECT_FALSE(!inp_cpy.lock());
	EXPECT_FALSE(!id_cpy.lock());
	EXPECT_FALSE(!pow_cpy.lock());

	// now we delete the topmost object
	pow.reset();

	// check whether the whole hierarchy has been destroyed
	EXPECT_TRUE(!inp_cpy.lock());
	EXPECT_TRUE(!id_cpy.lock());
	EXPECT_TRUE(!pow_cpy.lock());
}
