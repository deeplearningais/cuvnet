#include <cmath>
#include <fstream>
#include <cstdio>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include <cuv/basics/io.hpp>

#include <cuvnet/ops.hpp>
#include <cuvnet/op_io.hpp>

#include <boost/test/unit_test.hpp>

using namespace cuvnet;
using std::printf;

BOOST_AUTO_TEST_SUITE(op_test)

BOOST_AUTO_TEST_CASE(io){
	typedef boost::shared_ptr<Op> ptr_t;
	namespace bar= boost::archive;

	{
		ptr_t inp = boost::make_shared<ParameterInput>(cuv::extents[10][20]);
		ptr_t id  = boost::make_shared<Identity>(inp->result());

		std::ofstream f("test.ser");
		bar::binary_oarchive oa(f);
        register_objects(oa);
		oa << id;
	}

	std::ifstream f("test.ser");
	bar::binary_iarchive ia(f);
    register_objects(ia);
	ptr_t id;
	ia >> id;
	BOOST_CHECK(id != NULL);
	BOOST_CHECK(NULL != boost::dynamic_pointer_cast<Identity>(id));
	BOOST_CHECK_EQUAL(id->param()->param_uses.size(),1);
	ptr_t inp = id->param()->use(0)->get_op();
	BOOST_CHECK(NULL != boost::dynamic_pointer_cast<ParameterInput>(inp));
}

BOOST_AUTO_TEST_CASE(strio){
	typedef boost::shared_ptr<Op> ptr_t;
	namespace bar= boost::archive;

    std::string s;
	{
		ptr_t inp = boost::make_shared<ParameterInput>(cuv::extents[10][20]);
		ptr_t id  = boost::make_shared<Identity>(inp->result());

        s = op2str(id);
	}

    ptr_t id = str2op(s);
	BOOST_CHECK(NULL != id);
	BOOST_CHECK(NULL != boost::dynamic_pointer_cast<Identity>(id));
	BOOST_CHECK_EQUAL(id->param()->param_uses.size(),1);
	ptr_t inp = id->param()->use(0)->get_op();
	BOOST_CHECK(NULL != boost::dynamic_pointer_cast<ParameterInput>(inp));
}
BOOST_AUTO_TEST_SUITE_END()
