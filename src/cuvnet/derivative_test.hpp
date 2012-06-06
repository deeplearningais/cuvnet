#ifndef __DERIVATIVE_TEST_HPP__
#     define __DERIVATIVE_TEST_HPP__
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/sum.hpp>

#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#include <cassert>
#define EXPECT_TRUE(X) assert(X)
#define EXPECT_NEAR(X,Y,D) assert(((X)-(Y))*((X)-(Y))<((D)*(D)))
#endif


namespace cuvnet
{
    namespace derivative_testing
    {

#       define PM(X) cuvnet::derivative_testing::print(#X,X);
        void print(const std::string& s, const matrix& M);

        void set_delta_to_unit_vec(Op& o, unsigned int result, unsigned int i);

        unsigned int prod(const std::vector<unsigned int>& v);

#define derivative_tester_verbose(X, R) \
        std::cout << "Testing derivative of "<<#X<<":"<<std::endl; \
        cuvnet::derivative_testing::derivative_tester(X,R,true); \
        std::cout << "done."<<std::endl;
#define derivative_tester_verbose_prec(X, R, P) \
        std::cout << "Testing derivative of "<<#X<<":"<<std::endl; \
        cuvnet::derivative_testing::derivative_tester(X,R,true, P); \
        std::cout << "done."<<std::endl;

        void derivative_tester(Op& op, int result=0, bool verbose=false, double prec=0.003, float minv=1.0, float maxv=-1.0);

    } /* derivative_testing */
} /* cuvnet */
#endif /* __DERIVATIVE_TEST_HPP__ */
