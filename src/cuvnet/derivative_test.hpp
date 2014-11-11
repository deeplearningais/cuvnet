#ifndef __DERIVATIVE_TEST_HPP__
#     define __DERIVATIVE_TEST_HPP__
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/sum.hpp>


namespace cuvnet
{
    class swiper;
    namespace derivative_testing
    {

#       define PM(X) cuvnet::derivative_testing::print(#X,X);
        void print(const std::string& s, const matrix& M);

        void set_delta_to_unit_vec(Op& o, unsigned int result, unsigned int i);

        unsigned int prod(const std::vector<unsigned int>& v);

        /**
         * ensure that a function does not have a state
         * by executing it twice in succession
         * and checking whether results are equal.
         */
        void ensure_no_state(boost::shared_ptr<Sink> out, swiper& swp, const std::vector<Op*>& params);

        /**
         * determine output and all derivatives of an op, for comparison with another configuration.
         */
        std::vector<std::pair<std::string, cuv::tensor<float, cuv::host_memory_space> > >
            all_outcomes(boost::shared_ptr<Op> op);

#define derivative_tester_verbose(X, R) \
        std::cout << "Testing derivative of "<<#X<<":"<<std::endl; \
        cuvnet::derivative_testing::derivative_tester(X,R,true); \
        std::cout << "done."<<std::endl;
#define derivative_tester_verbose_prec(X, R, P) \
        std::cout << "Testing derivative of "<<#X<<":"<<std::endl; \
        cuvnet::derivative_testing::derivative_tester(X,R,true, P); \
        std::cout << "done."<<std::endl;

        //void derivative_tester(Op& op, int result=0, bool verbose=false, double prec=0.003, float minv=1.0, float maxv=-1.0);

        //namespace impl {
            struct derivative_tester{
                Op& m_op;
                int m_result;
                bool m_verbose;
                bool m_spread;
                std::string m_spread_filter;
                double m_prec;
                float m_minv, m_maxv;
                std::vector<Op*> m_derivable_params;
                bool m_simple_and_fast;
                unsigned int m_variant_filter;
                std::string m_parameter_filter;
                double m_epsilon;

                inline derivative_tester& precision(double d){m_prec = d; return *this;}
                inline derivative_tester& epsilon(double d){m_epsilon = d; return *this;}
                inline derivative_tester& result(int i){m_result = i; return *this;}
                inline derivative_tester& values(float minv, float maxv){m_minv = minv; m_maxv = maxv; return *this;}
                inline derivative_tester& verbose(bool verbose = true){m_verbose = verbose; return *this;}
                inline derivative_tester& full_jacobian(bool b = true){m_simple_and_fast = !b; return *this;}
                inline derivative_tester& only_param(std::string s){m_parameter_filter = s; return *this;}
                inline derivative_tester& only_variant(int filter){m_variant_filter = filter; return *this;}
                inline derivative_tester& without_variant(int filter){m_variant_filter &= ~filter; return *this;}
                inline derivative_tester& spread_values(bool b=true, std::string filter=""){m_spread = b; m_spread_filter=filter; return *this;}

                derivative_tester(Op& op);
                void test();
                
            private:

                // calls derivative_test_wrt
                void test_all(Op& op, int result, std::vector<Op*>& derivable_params, double prec, float minv, float maxv, bool spread, double epsilon);
                
                // tests the derivative of op w.r.t. pi.
                // (calculates /all/ params, but checks only one)
                void test_wrt(Op& op, int result, std::vector<Op*>& derivable_params, Op* raw, double prec, double epsilon);
            };
        //}

    } /* derivative_testing */
} /* cuvnet */
#endif /* __DERIVATIVE_TEST_HPP__ */
