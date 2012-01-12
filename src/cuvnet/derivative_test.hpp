#ifndef __DERIVATIVE_TEST_HPP__
#     define __DERIVATIVE_TEST_HPP__
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/output.hpp>

#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#include <cassert>
#define EXPECT_TRUE(X) assert(X)
#define EXPECT_NEAR(X,Y,D) assert(((X)-(Y))*((X)-(Y))<((D)*(D)))
#endif


namespace cuvnet
{
    namespace derivative_testing
    {

#       define PM(X) print(#X,X);
        void print(const std::string& s, const matrix& M){
            std::cout << "_________________________________________"<<std::endl;
            std::cout << "------------ "<<s<<" (";
            std::copy(M.shape().begin(), M.shape().end(),std::ostream_iterator<unsigned int>(std::cout,","));
            std::cout << ") ------------"<<std::endl;
            if(M.ndim()==1){
                unsigned int cnt=0;
                for (unsigned int i = 0; i < M.size(); ++i){
                    printf("% 2.5f ", (float)M[i]);
                    if((cnt++*9)>100){printf("\n");cnt=0;}
                }
            }
            if(M.ndim()==2){
                for (unsigned int i = 0; i < M.shape(0); ++i){
                    printf("   ");
                    unsigned int cnt=0;
                    for (unsigned int j = 0; j < M.shape(1); ++j){
                        printf("% 2.5f ", (float)M(i,j));
                        if((cnt++*9)>100){printf("\n");cnt=0;}
                    }
                    printf("\n");
                }
            }
        }

        void set_delta_to_unit_vec(Op::result_t& r, unsigned int i){
            r->delta.reset(new matrix(r->shape));
            r->delta.data()    = 0.f;
            r->delta.data()[i] = 1.f;
        }
        unsigned int prod(const std::vector<unsigned int>& v){
            return std::accumulate(v.begin(),v.end(),1u, std::multiplies<unsigned int>());
        }
#define derivative_tester_verbose(X) \
        std::cout << "Testing derivative of "<<#X<<":"<<std::endl; \
        cuvnet::derivative_testing::derivative_tester(X,true); \
        std::cout << "done."<<std::endl;

        void derivative_tester(Op& op, bool verbose=false, float prec=0.003f){
            // assumption: op has only one result
            boost::shared_ptr<Output> out_op = boost::make_shared<Output>(op.result());

            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            op.visit(pcv);

            // fill all params with random numbers
            BOOST_FOREACH(Op* raw, pcv.plist){
                Input* param = dynamic_cast<Input*>(raw);
                EXPECT_TRUE(param!=NULL);
                for (unsigned int i = 0; i < param->data().size(); ++i)
                {
                    param->data()[i] = 2.f;
                    //param->data()[i] = 2.f*(float)drand48()-1.0f;
                }
            }

            swiper swipe(op, true, pcv.plist);

            BOOST_FOREACH(Op* raw, pcv.plist){
                Input* param = dynamic_cast<Input*>(raw);
                std::cout << "...testing derivative w.r.t. "<<param->name()<<"..."<<std::endl;
                EXPECT_TRUE(param!=NULL);
                unsigned int n_inputs  = param->data().size();
                unsigned int n_outputs = prod(op.result()->shape);
                matrix J(n_outputs, n_inputs); J = 0.f;
                for(unsigned int out=0;out<n_outputs;out++){
                    swipe.fprop();
                    set_delta_to_unit_vec(op.result(),out);
                    swipe.bprop();

                    // set row in J to the backpropagated value
                    matrix d_in = param->result()->delta.cdata();
                    d_in.reshape(cuv::extents[n_inputs]);
                    matrix Jrow(cuv::indices[cuv::index_range(out,out+1)][cuv::index_range()], J);
                    Jrow = d_in;
                }

                matrix J_(n_inputs,n_outputs); J_ = 0.f;
                for (unsigned int in = 0; in < n_inputs; ++in) {
                    static const float eps = 0.00001f;
                    float v = param->data()[in];
                    param->data()[in] = v + eps;
                    swipe.fprop();
                    matrix o_plus     = out_op->cdata(); 
                    param->data()[in] = v - eps;
                    swipe.fprop();
                    matrix o_minus    = out_op->cdata();
                    param->data()[in] = v;

                    o_plus .reshape(cuv::extents[n_outputs]);
                    o_minus.reshape(cuv::extents[n_outputs]);
                    o_plus -= o_minus;
                    o_plus /= 2.f*eps;

                    // set row in J_ to finite-difference approximation
                    matrix J_row(cuv::indices[cuv::index_range(in,in+1)][cuv::index_range()], J_);
                    J_row = o_plus;
                }
                matrix J_t(n_outputs, n_inputs);
                cuv::transpose(J_t,J_);
                float testval = std::sqrt(cuv::norm2(J_t-J))/J.size();
                if(testval>prec){
                    PM(J_); PM(J);
                }
                EXPECT_NEAR(std::sqrt(cuv::norm2(J_t-J))/J.size(), 0.f, prec );
            }
        }
    } /* derivative_testing */
} /* cuvnet */
#endif /* __DERIVATIVE_TEST_HPP__ */
