#include "derivative_test.hpp"
#include <ext/functional>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/matwrite.hpp>
#include <cuvnet/tools/function.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/ops/axpby.hpp>
#include <cuvnet/ops/sum.hpp>

#include <boost/test/unit_test.hpp>

namespace {
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("derivative_tester"));
}

namespace cuvnet{ namespace derivative_testing {

    void ensure_no_state(boost::shared_ptr<Sink> out, swiper& swp, const std::vector<Op*>& params, bool verbose){
        Tracer t_top(g_log, "ensure_no_state");
        { // forward pass
            TRACE(g_log, "fprop");
            using namespace cuv;
            swp.fprop();
            tensor<float,host_memory_space> r0 = out->cdata().copy();
            swp.fprop();
            tensor<float,host_memory_space> r1 = out->cdata().copy();

            if(verbose)
                LOG4CXX_INFO(g_log, "-- Checking Results; cuv::minimum(r0):" << cuv::minimum(r0) << " cuv::maximum(r0):" << cuv::maximum(r0));

            BOOST_CHECK(out->result()->shape == r0.shape()); // shape should be as advertised by determine_shapes
            BOOST_CHECK(equal_shape(r0,r1));

            tensor<float,host_memory_space> rdiff(r0.shape());
            apply_binary_functor(rdiff, r0, r1, BF_SUBTRACT);
            apply_scalar_functor(rdiff, SF_ABS);
            BOOST_CHECK_LT(maximum(rdiff), 0.00000001);
        }
        Tracer t_bprop(g_log, "bprop");
        BOOST_FOREACH(Op* raw, params){
            using namespace cuv;
            ParameterInput* pi = dynamic_cast<ParameterInput*>(raw);
            TRACE(g_log, "param_" + pi->name())
            BOOST_CHECK(pi != NULL);
            pi->reset_delta();
            swp.fprop();
            swp.bprop();
            BOOST_CHECK(pi->data().shape() == pi->delta().shape());
            tensor<float,host_memory_space> r0 = pi->delta().copy();
            pi->reset_delta();
            swp.fprop();
            swp.bprop();
            tensor<float,host_memory_space> r1 = pi->delta().copy();
            pi->reset_delta();

            if(verbose)
                LOG4CXX_INFO(g_log, "-- Checking Gradients of `" << pi->name() << "'; cuv::minimum(r0):" << cuv::minimum(r0) << " cuv::maximum(r0):" << cuv::maximum(r0));

            BOOST_CHECK(equal_shape(r0,r1));
            tensor<float,host_memory_space> rdiff(r0.shape());
            apply_binary_functor(rdiff, r0, r1, BF_SUBTRACT);
            apply_scalar_functor(rdiff, SF_ABS);
            BOOST_CHECK_LT(maximum(rdiff), 0.00000001);
        }
    }

        void print(const std::string& s, const matrix& M){
            std::cout << "_________________________________________"<<std::endl;
            std::cout << "------------ "<<s<<" (";
            for(unsigned int s=0;s<(unsigned int)M.ndim();++s){
                std::cout << M.shape(s);
                if(s<(unsigned int)(M.ndim()-1))
                    std::cout << ", ";
            }
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

        void set_delta_to_unit_vec(Op& o, unsigned int result, unsigned int i){
            for(unsigned int ridx=0;ridx<o.get_n_results();ridx++){
                Op::result_t& r = o.result(ridx);
                r->delta.reset(new matrix(r->shape));
                r->delta.data()    = 0.f;
                if(ridx==result)
                    r->delta.data()[i] = 1.f;
            }
        }

        unsigned int prod(const std::vector<unsigned int>& v){
            return std::accumulate(v.begin(),v.end(),1u, std::multiplies<unsigned int>());
        }

        template<class ArgumentType, class ResultType>
        struct ptr_caster
        : public std::unary_function<ArgumentType*, ResultType*>
        {
            ResultType* operator()(ArgumentType*s)const{ return (ResultType*)s; }
        };

        void initialize_inputs(param_collector_visitor& pcv, float minv, float maxv){
            // fill all params with random numbers
            BOOST_FOREACH(Op* raw, pcv.plist){
                ParameterInput* param = dynamic_cast<ParameterInput*>(raw);
                BOOST_CHECK(param!=NULL);
                for (unsigned int i = 0; i < param->data().size(); ++i)
                {
                    //param->data()[i] = 2.f;
                    if(maxv>minv){
                        param->data()[i] = (float)((maxv-minv)*drand48()+minv);
                    }else if(maxv==minv){
                        // assume params are initialized already
                    }else{
                        param->data()[i] = (float)(0.1f + 0.9f*drand48()) * (drand48()<.5?-1.f:1.f); // avoid values around 0
                    }
                }
            }
        }

        void derivative_tester_impl(Op& op, int result, bool verbose, double prec, float minv, float maxv){
            // assumption: op has only one result
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op.result(result));

            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            op.visit(pcv);
            BOOST_CHECK(pcv.plist.size()>0);


            // find only those parameters which we can derive for
            // (the others are needed since they need to be initialized above)
            std::vector<Op*> derivable_params;
            std::remove_copy_if(     // think of it as "copy except"
                    pcv.plist.begin(), pcv.plist.end(),
                    std::back_inserter(derivable_params),
                    std::not1( // not [  cast_to_input(op)->derivable()   ]
                        __gnu_cxx::compose1(
                            std::mem_fun( &ParameterInput::derivable ),
                            ptr_caster<Op,ParameterInput>()))
                    );
            BOOST_CHECK(derivable_params.size() > 0);

            swiper swipe(op, result, derivable_params);
            swipe.dump("derivative_tester.dot", true);

            initialize_inputs(pcv, minv, maxv);
            {
                if(verbose)
                    LOG4CXX_INFO(g_log, "  -ensuring function is stateless");
                boost::shared_ptr<Op> p = op.shared_from_this();
                ensure_no_state(out_op, swipe, derivable_params, verbose);
            }


            BOOST_FOREACH(Op* raw, derivable_params){
                ParameterInput* param = dynamic_cast<ParameterInput*>(raw);
                BOOST_CHECK(param!=NULL);
                if(!param->derivable())
                    continue;
                unsigned int n_inputs  = param->data().size();
                unsigned int n_outputs = prod(op.result(result)->shape);
                matrix J(n_outputs, n_inputs); J = 0.f;
                TRACE(g_log, "wrt_" + param->name());
                if(verbose)
                {
                    LOG4CXX_INFO(g_log, "  -testing derivative w.r.t. "<<param->name());
                    LOG4CXX_INFO(g_log, "   Jacobi dims: "<<n_outputs<<" x "<<n_inputs);
                }
                for(unsigned int out=0;out<n_outputs;out++){
                    swipe.fprop();
                    cuvAssert(!cuv::has_nan(out_op->cdata()));
                    cuvAssert(!cuv::has_inf(out_op->cdata()));
                    set_delta_to_unit_vec(op,result,out);
                    param->reset_delta();
                    swipe.bprop(false);

                    // set row in J to the backpropagated value
                    matrix d_in = param->delta();
                    cuvAssert(!cuv::has_nan(d_in));
                    cuvAssert(!cuv::has_inf(d_in));
                    d_in.reshape(cuv::extents[n_inputs]);
                    J[cuv::indices[cuv::index_range(out,out+1)][cuv::index_range()]] = d_in;
                }
                cuv::tensor<float,cuv::host_memory_space> Jh = J; J.dealloc(); // save device space

                matrix J_(n_inputs,n_outputs); J_ = 0.f;
                for (unsigned int in = 0; in < n_inputs; ++in) {
                    static const double eps = 0.001;
                    float v = param->data()[in];
                    param->data()[in] = (float)((double)v + eps);
                    swipe.fprop();
                    matrix o_plus     = out_op->cdata().copy();
                    param->data()[in] = (float)((double)v - eps);
                    swipe.fprop();
                    matrix o_minus    = out_op->cdata().copy();
                    param->data()[in] = v;

                    o_plus .reshape(cuv::extents[n_outputs]);
                    o_minus.reshape(cuv::extents[n_outputs]);
                    o_plus -= o_minus;
                    o_plus /= (float)(2.0*eps);

                    // set row in J_ to finite-difference approximation
                    J_[cuv::indices[cuv::index_range(in,in+1)][cuv::index_range()]] = o_plus;
                }
                cuv::tensor<float,cuv::host_memory_space> J_h = J_; J_.dealloc(); // save device space
                cuv::tensor<float,cuv::host_memory_space> J_t(n_outputs, n_inputs);
                cuv::transpose(J_t,J_h); J_h.dealloc();
                cuv::tensor<float, cuv::host_memory_space> tmp(Jh.shape());
                if(verbose)
                {
                    LOG4CXX_INFO(g_log, "   range(Jh)[finite differences]="<<cuv::maximum(Jh)-cuv::minimum(Jh));
                    LOG4CXX_INFO(g_log, "   range(J_t)[analytical]       ="<<cuv::maximum(J_t)-cuv::minimum(J_t));
                }
                cuv::apply_binary_functor(tmp, J_t, Jh, cuv::BF_SUBTRACT);
                cuv::apply_scalar_functor(tmp, cuv::SF_SQUARE);
                double maxdiff = cuv::maximum(tmp);    // squared(!)
                double prec_  = prec * prec;                       // square precision, too
                if(verbose)
                {
                    LOG4CXX_INFO(g_log, "   maxdiff="<<maxdiff<<", prec_="<<prec_);
                    LOG4CXX_INFO(g_log, "   range(differences)="<<cuv::maximum(tmp)-cuv::minimum(tmp));
                }
                if(maxdiff>prec_){
                    LOG4CXX_WARN(g_log, "   maxdiff="<<maxdiff<<", prec_="<<prec_ << " dumping Jacobi matrices (analyticalJ.npy, finitediffJ.npy)");
                    tofile("analyticalJ.npy", Jh);
                    tofile("finitediffJ.npy", J_t);
                }
                BOOST_CHECK_LT(maxdiff, prec_ );
            }
        }

        void derivative_tester(Op& op, int result, bool verbose, double prec, float minv, float maxv){
            determine_shapes(op);
            bool simple_and_fast = true;
            std::vector<unsigned int> shape = op.result(result)->shape;
            boost::shared_ptr<ParameterInput> otherin = boost::make_shared<ParameterInput>(shape, "dummy_input");
            unsigned int factor = std::accumulate(shape.begin(), shape.end(), 1u, std::multiplies<unsigned int>());
            factor = std::min(factor, 10u); // give it some leeway in case we're summing over outputs.
            otherin->set_derivable(false);
            {
                TRACE(g_log, "plain");
                if(!simple_and_fast)
                    derivative_tester_impl(op, result, verbose, prec, minv, maxv);
                else{
                    boost::shared_ptr<Sum> func = boost::make_shared<Sum>(op.result(result));
                    derivative_tester_impl(*func, 0, verbose, prec * factor, minv, maxv);
                }
            }
            {
                TRACE(g_log, "variant_a");
                //boost::shared_ptr<Op> func = boost::make_shared<Axpby>(otherin->result(), op.result(result), 2.f, 2.f);
                boost::shared_ptr<Sum> func2 = boost::make_shared<Sum>(op.result(0));
                add_to_param(func2, otherin);
                derivative_tester_impl(*func2, 0, verbose, prec * factor, minv, maxv);
            }
            {
                TRACE(g_log, "variant_b");
                //boost::shared_ptr<Op> func = boost::make_shared<Axpby>(op.result(result), otherin->result(), 2.f, 2.f);
                boost::shared_ptr<Sum> func2 = boost::make_shared<Sum>(otherin->result(0));
                add_to_param(func2, op.shared_from_this());
                derivative_tester_impl(*func2, 0, verbose, prec * factor, minv, maxv);
            }
            {
                // TODO put something /before/ the op
                TRACE(g_log, "variant_c");
                //boost::shared_ptr<Op> func = boost::make_shared<Axpby>(otherin->result(), op.result(result), 2.f, 2.f);
                boost::shared_ptr<Sum> func2 = boost::make_shared<Sum>(op.result(0));
                boost::shared_ptr<Sum> func3 = boost::make_shared<Sum>(op.result(0));
                boost::shared_ptr<Axpby> func4 = boost::make_shared<Axpby>(func2->result(0), func3->result(0));

                derivative_tester_impl(*func4, 0, verbose, prec * factor, minv, maxv);
            }
        }
} }
