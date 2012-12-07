// vim:ts=4:sw=4:et:
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <cstdio>

#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>

#include <cuvnet/ops/axpby.hpp>
#include <cuvnet/ops/identity.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/mat_plus_vec.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/pow.hpp>
#include <cuvnet/ops/prod.hpp>

#include <boost/test/unit_test.hpp>

using namespace cuvnet;
using std::printf;


BOOST_AUTO_TEST_SUITE(op_test)
BOOST_AUTO_TEST_CASE(wiring){
	typedef boost::shared_ptr<Op> ptr_t;
    ptr_t inp = boost::make_shared<ParameterInput>(cuv::extents[10][20]);
    ptr_t id  = boost::make_shared<Identity>(inp->result());

    // find input object
    param_collector_visitor pcv;
    id->visit(pcv);
    BOOST_CHECK_EQUAL(pcv.plist.size(),1);

    // determine which derivatives to calculate
    BOOST_CHECK_EQUAL(id->param(0)->need_derivative, false);
    id->set_calculate_derivative(pcv.plist);
    BOOST_CHECK_EQUAL(id->param(0)->need_derivative, true);

    // shape propagation
    determine_shapes(*id);
    BOOST_CHECK_EQUAL(*pcv.plist.begin(),inp.get());
    BOOST_CHECK_EQUAL(id->result(0)->shape.at(0), 10);
    BOOST_CHECK_EQUAL(id->result(0)->shape.at(1), 20);
}

BOOST_AUTO_TEST_CASE(fprop_and_bprop){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[10][20]);
    ptr_t pow                     = boost::make_shared<Pow>(2,inp->result());
    boost::shared_ptr<Sink> out = boost::make_shared<Sink>(pow->result());

    boost::dynamic_pointer_cast<ParameterInput>(inp)->data() = 2.f;

    // tell that we want derivative w.r.t. all params
    param_collector_visitor pcv;
    pow->visit(pcv);
    BOOST_CHECK_EQUAL(pcv.plist.size(),1);
    pow->set_calculate_derivative(pcv.plist);
    BOOST_CHECK_EQUAL(pow->param(0)->need_derivative, true);
    determine_shapes(*pow);
    BOOST_CHECK_EQUAL(pow->result()->shape[0],10);
    BOOST_CHECK_EQUAL(pow->result()->shape[1],20);

    // manually tell which results we need
    inp->result(0)->need_result = true;
    pow->need_result(true);
    pow->result(0)->need_result = true;
    
    // manually tell which derivatives we need
    inp->need_derivative(true);
    pow->need_derivative(true);
    pow->param()->need_derivative = true;


    inp->fprop();
    pow->fprop();
    out->fprop();
    BOOST_CHECK_EQUAL(out->cdata()[0],4);

    pow->result()->delta.reset(new matrix(pow->result()->shape));
    pow->result()->delta.data() = 1.f;
    pow->bprop();
    inp->bprop();
    BOOST_CHECK_EQUAL(inp->delta()[0],4);
}

BOOST_AUTO_TEST_CASE(toposort){
    typedef boost::shared_ptr<Op> ptr_t;

    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[8][3]);
    ptr_t func0		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
    ptr_t func1		       = boost::make_shared<Prod>(inp0->result(), inp1->result(),'t','t');
    ptr_t func 		       = boost::make_shared<Axpby>(func0->result(), func1->result(), 1.3,1.5);

    // tell that we want derivative w.r.t. all params
    param_collector_visitor pcv;
    func->visit(pcv);
    BOOST_CHECK_EQUAL(pcv.plist.size(),2);
    func->set_calculate_derivative(pcv.plist);

    // determine sequence in which to call fwd, bwd pass
    toposort_visitor tv;
    func->visit(tv);

    BOOST_CHECK_EQUAL(tv.plist.size(), 5);
    BOOST_CHECK(tv.plist.at(0) == inp0.get() || tv.plist.at(0) == inp1.get());
    BOOST_CHECK(tv.plist.at(1) == inp0.get() || tv.plist.at(1) == inp1.get());
    BOOST_CHECK(tv.plist.at(2) == func0.get() || tv.plist.at(2) == func1.get());
    BOOST_CHECK(tv.plist.at(3) == func0.get() || tv.plist.at(3) == func1.get());
    BOOST_CHECK_EQUAL(tv.plist.at(4), func.get());
}


BOOST_AUTO_TEST_CASE(destruction){
	typedef boost::shared_ptr<Op> ptr_t;
	boost::shared_ptr<ParameterInput>  inp = boost::make_shared<ParameterInput>(cuv::extents[10][20]);
	ptr_t id                      = boost::make_shared<Identity>(inp->result());
	ptr_t pow                     = boost::make_shared<Pow>(2,id->result());

	boost::weak_ptr<Op> inp_cpy(inp);
	boost::weak_ptr<Op> id_cpy(id);
	boost::weak_ptr<Op> pow_cpy(pow);

	BOOST_CHECK(inp_cpy.lock() != NULL);
	BOOST_CHECK(id_cpy.lock() != NULL);
	BOOST_CHECK(pow_cpy.lock() != NULL);

	inp.reset();
	id.reset();

	// hierarchy should /still/ exist!
	BOOST_CHECK(NULL != inp_cpy.lock());
	BOOST_CHECK(NULL != id_cpy.lock());
	BOOST_CHECK(NULL != pow_cpy.lock());

	// now we delete the topmost object
	pow.reset();

	// check whether the whole hierarchy has been destroyed
	BOOST_CHECK(!inp_cpy.lock());
	BOOST_CHECK(!id_cpy.lock());
	BOOST_CHECK(!pow_cpy.lock());
}

BOOST_AUTO_TEST_CASE(t_write_graphviz){
	typedef boost::shared_ptr<Op> ptr_t;
    boost::shared_ptr<ParameterInput>  inp0 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    boost::shared_ptr<ParameterInput>  inp1 = boost::make_shared<ParameterInput>(cuv::extents[3][5]);
    ptr_t func                     = boost::make_shared<Axpby>(inp0->result(), inp1->result(), 1.3, -2.5);
    func                           = boost::make_shared<Pow>(2.f,func->result());
    std::ofstream os("test.dot");
    write_graphviz(*func,os);
}
BOOST_AUTO_TEST_SUITE_END()
