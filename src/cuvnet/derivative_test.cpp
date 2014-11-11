#include <boost/regex.hpp>
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
    log4cxx::LoggerPtr g_enslog(log4cxx::Logger::getLogger("ensure_no_state"));
}

namespace cuvnet{ namespace derivative_testing {

    std::vector<cuv::tensor<float, cuv::host_memory_space> >
        all_outcomes(boost::shared_ptr<Op> op){
            std::vector<cuv::tensor<float, cuv::host_memory_space> > results;
            cuvnet::function f(op, 0);
            cuv::tensor<float, cuv::host_memory_space> hr(f.evaluate());
            results.push_back(hr);
            param_collector_visitor pcv;
            op->visit(pcv);
            std::vector<Op*> params;
            for (unsigned int i = 0; i < pcv.plist.size(); i++) {
                if (!((ParameterInput*) pcv.plist[i])->derivable())
                    continue;
                params.push_back(pcv.plist[i]);
            }
            swiper swipe(*op, 0, params);
            swipe.fprop();
            swipe.bprop();
            for (unsigned int i = 0; i < params.size(); i++) {
                cuv::tensor<float, cuv::host_memory_space> hr(((ParameterInput*)params[i])->delta());
                results.push_back(hr);
            }
            return results;
        }

    void ensure_no_state(boost::shared_ptr<Sink> out, swiper& swp, const std::vector<Op*>& params, bool verbose){
        double epsilon = 0.00000001;
        { TRACE(g_enslog, "fprop");
            using namespace cuv;
            {
                TRACE(g_enslog, "A");
                swp.fprop();
            }
            tensor<float,host_memory_space> r0 = out->cdata().copy();
            {
                TRACE(g_enslog, "B");
                swp.fprop();
            }
            tensor<float,host_memory_space> r1 = out->cdata().copy();

            if(verbose)
                LOG4CXX_INFO(g_enslog, "-- Checking Results; cuv::minimum(r0):" << cuv::minimum(r0) << " cuv::maximum(r0):" << cuv::maximum(r0));

            BOOST_CHECK(out->result()->shape == r0.shape()); // shape should be as advertised by determine_shapes
            BOOST_CHECK(equal_shape(r0,r1));

            tensor<float,host_memory_space> rdiff(r0.shape());
            apply_binary_functor(rdiff, r0, r1, BF_SUBTRACT);
            apply_scalar_functor(rdiff, SF_ABS);
            double fprop_error = maximum(rdiff);
            if(fprop_error > epsilon){
                LOG4CXX_ERROR(g_enslog, " fprop error greater epsilon: "<< fprop_error);
            }

            BOOST_CHECK_LT(fprop_error, epsilon);
        }
        Tracer t_bprop(g_enslog, "bprop");
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
                LOG4CXX_INFO(g_enslog, "-- Checking Gradients of `" << pi->name() << "'; cuv::minimum(r0):" << cuv::minimum(r0) << " cuv::maximum(r0):" << cuv::maximum(r0));

            BOOST_CHECK(equal_shape(r0,r1));
            tensor<float,host_memory_space> rdiff(r0.shape());
            apply_binary_functor(rdiff, r0, r1, BF_SUBTRACT);
            apply_scalar_functor(rdiff, SF_ABS);
            double bprop_error = maximum(rdiff);
            if(bprop_error > epsilon){
                LOG4CXX_ERROR(g_enslog, " bprop error greater epsilon: "<< bprop_error);
            }
            BOOST_CHECK_LT(bprop_error, epsilon);
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

        void initialize_inputs(param_collector_visitor& pcv, float minv, float maxv, bool spread, std::string spread_filter){
            // fill all params with random numbers
            BOOST_FOREACH(Op* raw, pcv.plist){
                ParameterInput* param = dynamic_cast<ParameterInput*>(raw);
                if(spread){
                    bool matches = true;
                    if(spread_filter.size()){
                        boost::regex e(spread_filter);
                        if(!boost::regex_match(param->name(), e))
                            matches = false;
                    }
                    if(matches){
                        if(maxv != minv){
                            cuv::tensor<float, cuv::host_memory_space> t = param->data();
                            cuv::sequence(t);
                            std::random_shuffle(t.ptr(), t.ptr() + t.size());
                            t /= (float) t.size();
                            t *= maxv-minv;
                            t += minv;
                            param->data() = t;
                        }
                        continue;
                    }
                }

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

        derivative_tester::derivative_tester(Op& op)
            :m_op(op)
            ,m_result(0)
            ,m_verbose(false)
            ,m_spread(false)
            ,m_prec(0.01)
            ,m_minv(-1.)
            ,m_maxv(1.)
            ,m_simple_and_fast(true)
            ,m_variant_filter(~0)
            ,m_epsilon(0.001)
        {
            // tell that we want derivative w.r.t. all params
            param_collector_visitor pcv;
            m_op.visit(pcv);
            BOOST_CHECK(pcv.plist.size()>0);

            // find only those parameters which we can derive for
            // (the others are needed since they need to be initialized above)
            //std::vector<Op*> derivable_params;
            std::remove_copy_if(     // think of it as "copy except"
                    pcv.plist.begin(), pcv.plist.end(),
                    std::back_inserter(m_derivable_params),
                    std::not1( // not [  cast_to_input(op)->derivable()   ]
                        __gnu_cxx::compose1(
                            std::mem_fun( &ParameterInput::derivable ),
                            ptr_caster<Op,ParameterInput>()))
                    );
            BOOST_CHECK(m_derivable_params.size() > 0);
        }

        void derivative_tester::test() {
            determine_shapes(m_op);
            
            std::vector<unsigned int> shape = m_op.result(m_result)->shape;
            boost::shared_ptr<ParameterInput> otherin = boost::make_shared<ParameterInput>(shape, "dummy_input");
            unsigned int factor = std::accumulate(shape.begin(), shape.end(), 1u, std::multiplies<unsigned int>());
            factor = std::min(factor, 10u); // give it some leeway in case we're summing over outputs.
            otherin->set_derivable(false);
            if(m_variant_filter & 1){
                TRACE(g_log, "plain");
                if(!m_simple_and_fast){
                    test_all(m_op, m_result, m_derivable_params, m_prec, m_minv, m_maxv, m_spread, m_epsilon);
                } else {
                    boost::shared_ptr<Op> func = boost::make_shared<Sum>(m_op.result(m_result));
                    func = label("variant_plain", func);
                    test_all(*func, 0, m_derivable_params, m_prec * factor, m_minv, m_maxv, m_spread, m_epsilon);
                }
            }
            if(m_variant_filter & 2){
                TRACE(g_log, "variant_a");
                
                boost::shared_ptr<Op> func2;
                if(!m_simple_and_fast)
                    func2 = boost::make_shared<AddScalar>(m_op.result(m_result),1.f);
                else
                    func2 = boost::make_shared<Sum>(m_op.result(m_result));
                add_to_param(func2, otherin);
                func2 = label("variant_a", func2);
                test_all(*func2, 0, m_derivable_params, m_prec, m_minv, m_maxv, m_spread, m_epsilon);
            }
            if(m_variant_filter & 4){
                TRACE(g_log, "variant_b");
                
                boost::shared_ptr<Op> func;
                if(!m_simple_and_fast)
                    func = boost::make_shared<AddScalar>(otherin->result(0), 1.f);
                else
                    func = boost::make_shared<Sum>(otherin->result(0));
                add_to_param(func, m_op.shared_from_this());
                func = label("variant_b", func);
                test_all(*func, 0, m_derivable_params, m_prec, m_minv, m_maxv, m_spread, m_epsilon);
            }
            if(m_variant_filter & 8){
                TRACE(g_log, "variant_c");
                
                boost::shared_ptr<Op> func1, func2;
                if (!m_simple_and_fast){
                    func1 = boost::make_shared<AddScalar>(m_op.result(m_result), 1.f);
                    func2 = boost::make_shared<AddScalar>(m_op.result(m_result), 1.f);
                } else {
                    func1 = boost::make_shared<Sum>(m_op.result(m_result));
                    func2 = boost::make_shared<Sum>(m_op.result(m_result));
                }

                boost::shared_ptr<Op> func3 = boost::make_shared<Axpby>(func1->result(0), func2->result(0));

                func3 = label("variant_c", func3);
                test_all(*func3, 0, m_derivable_params, m_prec, m_minv, m_maxv, m_spread, m_epsilon);
            }
            if(m_variant_filter & 16){
                param_collector_visitor pcv;
                m_op.visit(pcv);
                BOOST_CHECK(pcv.plist.size()>0);
                for (unsigned int i = 0; i < pcv.plist.size(); i++) {
                    if (((ParameterInput*) pcv.plist[i])->derivable()) {
                        TRACE(g_log, "variant_d" + boost::lexical_cast<std::string>(i));
                        boost::shared_ptr<Sum> func1 = boost::make_shared<Sum>(pcv.plist[i]->result());
                       
                        std::swap(pcv.plist[i]->result()->result_uses.front(), pcv.plist[i]->result()->result_uses.back());
                        boost::shared_ptr<Op> func2 = boost::make_shared<Axpby>(m_op.result(m_result), func1->result());
                        if (m_simple_and_fast)
                            func2 = boost::make_shared<Sum>(func2->result());
                    
                        func2 = label("variant_d" + boost::lexical_cast<std::string>(i), func2);
                        
                        test_all(*func2, 0, m_derivable_params, m_prec, m_minv, m_maxv, m_spread, m_epsilon);
                        
                        std::swap(pcv.plist[i]->result()->result_uses.front(), pcv.plist[i]->result()->result_uses.back());
                    }
                }
            }
        }

        void derivative_tester::test_all(Op& op, int result, std::vector<Op*>& derivable_params, double prec, float minv, float maxv, bool spread, double epsilon) {
            // assumption: op has only one result
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op.result(result));

            swiper swipe(op, result, derivable_params);
            if(m_verbose)
                swipe.dump("derivative_tester.dot", true);

            param_collector_visitor pcv;
            op.visit(pcv);
            initialize_inputs(pcv, minv, maxv, spread, m_spread_filter);
            {
                if(m_verbose)
                    LOG4CXX_INFO(g_log, "  -ensuring function is stateless");
                boost::shared_ptr<Op> p = op.shared_from_this();
                ensure_no_state(out_op, swipe, derivable_params, m_verbose);
            }


            //BOOST_FOREACH(Op* raw, m_derivable_params){
            for (unsigned int i = 0; i < derivable_params.size(); i++) {
                Op* raw = derivable_params[i];
                ParameterInput* pi = dynamic_cast<ParameterInput*>(raw);
                if(pi && m_parameter_filter.size()){
                    boost::regex e(m_parameter_filter);
                    if(!boost::regex_match(pi->name(), e))
                        continue;
                }
                std::swap(derivable_params[i], derivable_params[0]);
                
                //ParameterInput* pi = dynamic_cast<ParameterInput*>(raw);
                //BOOST_CHECK(pi != NULL);
                test_wrt(op, result, derivable_params, raw, prec, epsilon);
                
                std::swap(derivable_params[i], derivable_params[0]);
            }
        }
        
        void derivative_tester::test_wrt(Op& op, int result, std::vector<Op*>& derivable_params, Op* raw, double prec, double eps){
            boost::shared_ptr<Sink> out_op = boost::make_shared<Sink>(op.result(result));
            
            ParameterInput* param = dynamic_cast<ParameterInput*>(raw);
            
            swiper swipe(op, result, derivable_params);
            swipe.dump("derivative_tester_wrt_"+param->name()+".dot", true);

            BOOST_CHECK(param != NULL);
            if(!param->derivable())
                return; // why is this necessary?
            unsigned int n_inputs  = param->data().size();
            unsigned int n_outputs = prod(op.result(result)->shape);
            matrix J(n_outputs, n_inputs); J = 0.f;
            TRACE(g_log, "wrt_" + param->name());
            if(m_verbose)
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
            if(m_verbose)
            {
                LOG4CXX_INFO(g_log, "   range(Jh)[analytical]          ="<<cuv::maximum(Jh )-cuv::minimum(Jh )<<" min:"<<cuv::minimum(Jh )<<" max:"<<cuv::maximum(Jh ));
                LOG4CXX_INFO(g_log, "   range(J_t)[finite differences] ="<<cuv::maximum(J_t)-cuv::minimum(J_t)<<" min:"<<cuv::minimum(J_t)<<" max:"<<cuv::maximum(J_t));
            }
            cuv::apply_binary_functor(tmp, J_t, Jh, cuv::BF_SUBTRACT);
            LOG4CXX_INFO(g_log, "   range(diff) ="<<cuv::maximum(tmp)-cuv::minimum(tmp));
            {
                // normalize error by dividing by either min, max or max-min, 
                // whichever provides the largest absolute value
                float _min = std::abs(cuv::minimum(J_t));
                float _max = std::abs(cuv::maximum(J_t));
                float _rng = std::abs(cuv::maximum(J_t) - cuv::minimum(J_t));
                float fact = std::max(_min, std::max(_max, _rng)) + 0.001f;
                tmp /= fact;
                LOG4CXX_INFO(g_log, "   diff factor =" << fact);
            }
            //cuv::apply_scalar_functor(tmp, cuv::SF_SQUARE);
            //double maxdiff = cuv::maximum(tmp);    // squared(!)
            double maxdiff = std::max(cuv::maximum(tmp), std::abs(cuv::minimum(tmp)));    // squared(!)
            //double prec_  = prec * prec;                       // square precision, too
            double prec_  = prec;                       // square precision, too
            if(m_verbose)
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
} }
