#include "derivative_test.hpp"
#include <ext/functional>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/matwrite.hpp>
#include <cuvnet/tools/function.hpp>

#include <boost/test/unit_test.hpp>

namespace cuvnet{ namespace derivative_testing {

    void ensure_no_state(boost::shared_ptr<Sink> out, swiper& swp, const std::vector<Op*>& params){
        { // forward pass
            swp.fprop();
            cuv::tensor<float,cuv::host_memory_space> r0 = out->cdata().copy();
            swp.fprop();
            cuv::tensor<float,cuv::host_memory_space> r1 = out->cdata().copy();

            BOOST_CHECK(cuv::equal_shape(r0,r1));
            cuv::tensor<float,cuv::host_memory_space> rdiff(r0-r1);
            cuv::apply_scalar_functor(rdiff, cuv::SF_ABS);
            BOOST_CHECK_LT(cuv::maximum(rdiff), 0.00000001);
        }
        BOOST_FOREACH(Op* raw, params){
            ParameterInput* pi = dynamic_cast<ParameterInput*>(raw);
            BOOST_CHECK(pi != NULL);
            pi->reset_delta();
            swp.fprop();
            swp.bprop();
            cuv::tensor<float,cuv::host_memory_space> r0 = pi->delta().copy();
            pi->reset_delta();
            swp.fprop();
            swp.bprop();
            cuv::tensor<float,cuv::host_memory_space> r1 = pi->delta().copy();
            pi->reset_delta();

            BOOST_CHECK(cuv::equal_shape(r0,r1));
            cuv::tensor<float,cuv::host_memory_space> rdiff(r0-r1);
            cuv::apply_scalar_functor(rdiff, cuv::SF_ABS);
            BOOST_CHECK_LT(cuv::maximum(rdiff), 0.00000001);
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

        void derivative_tester(Op& op, int result, bool verbose, double prec, float minv, float maxv){
            // assumption: op has only one result
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op.result(result));

            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            op.visit(pcv);
            BOOST_CHECK(pcv.plist.size()>0);

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

            {
                if(verbose)
                    std::cout << "  -ensuring function is stateless" << std::endl;
                boost::shared_ptr<Op> p = op.shared_from_this();
                ensure_no_state(out_op, swipe, derivable_params);
            }


            BOOST_FOREACH(Op* raw, derivable_params){
                ParameterInput* param = dynamic_cast<ParameterInput*>(raw);
                BOOST_CHECK(param!=NULL);
                if(!param->derivable())
                    continue;
                unsigned int n_inputs  = param->data().size();
                unsigned int n_outputs = prod(op.result(result)->shape);
                matrix J(n_outputs, n_inputs); J = 0.f;
                if(verbose)
                {
                std::cout << "  -testing derivative w.r.t. "<<param->name()<<""<<std::endl;
                std::cout << "   Jacobi dims: "<<n_outputs<<" x "<<n_inputs<<"..."<<std::endl;
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
                double maxdiff = cuv::maximum((J_t-Jh)*(J_t-Jh));    // squared(!) 
                double prec_  = prec * prec;                       // square precision, too
                if(verbose)
                {
                    std::cout << "   maxdiff="<<maxdiff<<", prec_="<<prec_<<std::endl;
                    std::cout << "   range(Jh)="<<cuv::maximum(Jh)-cuv::minimum(Jh)<<std::endl;
                }
                if(maxdiff>prec_){
                    std::cout << "   maxdiff="<<maxdiff<<", prec_="<<prec_<<std::endl;
                    std::cout << "   dumping Jacobi matrices: " << std::endl;
                    std::cout << "   - analyticalJ.npy" << std::endl;
                    std::cout << "   - finitediffJ.npy" << std::endl;
                    tofile("analyticalJ.npy", Jh);
                    tofile("finitediffJ.npy", J_t);
                }
                BOOST_CHECK_LT(maxdiff, prec_ );
            }
        }

        
} }
