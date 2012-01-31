#include <cmath>
#include <fstream>
#include <cstdio>
#include <gtest/gtest.h>


#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include <cuv/basics/io.hpp>

#include <cuvnet/ops.hpp>
#include <cuvnet/op_io.hpp>

using namespace cuvnet;
using std::printf;

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

TEST(op_test,strio){
	typedef boost::shared_ptr<Op> ptr_t;
	namespace bar= boost::archive;

    std::string s;
	{
		ptr_t inp = boost::make_shared<Input>(cuv::extents[10][20]);
		ptr_t id  = boost::make_shared<Identity>(inp->result());

        s = op2str(id);
	}

    ptr_t id = str2op(s);
	EXPECT_FALSE(!id);
	EXPECT_FALSE(!boost::dynamic_pointer_cast<Identity>(id));
	EXPECT_EQ(id->param()->param_uses.size(),1);
	ptr_t inp = id->param()->use(0)->get_op();
	EXPECT_FALSE(!boost::dynamic_pointer_cast<Input>(inp));
}
