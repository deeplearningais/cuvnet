#include <cmath>
#include <fstream>
#include <cstdio>
#include <gtest/gtest.h>

#include <cuvnet/op.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include <cuv/basics/io.hpp>

using namespace cuvnet;
using std::printf;

BOOST_CLASS_EXPORT_GUID(Op,       "Op");
BOOST_CLASS_EXPORT_GUID(Input,    "Input");
BOOST_CLASS_EXPORT_GUID(Output,   "Output");
BOOST_CLASS_EXPORT_GUID(Identity, "Identity");
BOOST_CLASS_EXPORT_GUID(Pow,      "Pow");
BOOST_CLASS_EXPORT_GUID(Tanh,     "Tanh");



TEST(op_test,io){
	typedef boost::shared_ptr<Op> ptr_t;
	namespace bar= boost::archive;

	{
		ptr_t inp = boost::make_shared<Input>(cuv::extents[10][20]);
		ptr_t id  = boost::make_shared<Identity>(inp->result());

		std::ofstream f("test.ser");
		bar::binary_oarchive oa(f);
		oa << id;
	}

	std::ifstream f("test.ser");
	bar::binary_iarchive ia(f);
	ptr_t id;
	ia >> id;
	EXPECT_FALSE(!id);
	EXPECT_FALSE(!boost::dynamic_pointer_cast<Identity>(id));
	EXPECT_EQ(id->param()->param_uses.size(),1);
	ptr_t inp = id->param()->use(0)->get_op();
	EXPECT_FALSE(!boost::dynamic_pointer_cast<Input>(inp));
}
