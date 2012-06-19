#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>
//#include <glog/logging.h>

#include <cuvnet/op.hpp>

using namespace cuvnet;

TEST(cow_ptr_test, init){
	cow_ptr<int> k;
	EXPECT_TRUE(!k);

	cow_ptr<int> i(new int(4));
	EXPECT_EQ(i.cdata(),4);

	// copying ptr should result in copy of data
	cow_ptr<int> j = i;
	EXPECT_EQ(&i.cdata(), &j.cdata());

	// writing to ptr should result in copy of data
	*i = 3;
	EXPECT_NE(&i.cdata(), &j.cdata());
	EXPECT_EQ(i.cdata(), 3);

	// changing the only ptr to a value should not result in reallocation
	const int* old = &i.cdata();
	*i = 4;
	EXPECT_EQ(&i.cdata(), old);
}

