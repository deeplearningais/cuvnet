#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <gtest/gtest.h>
#include <tools/logging.hpp>

#include <cuvnet/op.hpp>

using namespace cuvnet;

TEST(cuvnetlog, init){
	Logger lg(std::cout);
	lg.log(0)<<bson_pair("foo","bar");
	auto x = lg.log(0);
    x <<bson_pair("foo","bar");
    x <<bson_pair("foo2", 5.6);
}
