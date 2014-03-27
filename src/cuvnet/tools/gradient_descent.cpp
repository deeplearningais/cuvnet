#include <iomanip>
#include<boost/limits.hpp>
#include<cuv/tensor_ops/rprop.hpp>
#include<cuv/tensor_ops/spn_gd.hpp>
#include "gradient_descent.hpp"
#include <log4cxx/logger.h>
#include <log4cxx/mdc.h>
#include <log4cxx/ndc.h>
#include <cuv/tools/device_tools.hpp>
#include <cuv/libs/opt/opt.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <iostream>
#include <fstream>
#include <cuvnet/tools/matwrite.hpp>

namespace cuvnet
{
gradient_descent::gradient_descent(const Op::op_ptr& op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay)
    : m_result(result), m_params(params), m_learnrate(learnrate), /*m_learnrate_decay(1.f), */m_weightdecay(weightdecay)
        , m_epoch(0), m_epoch_of_saved_params(0)
        , m_swipe(*op,result,params,false), m_update_every(1)
{ 
    // remove doublets
    std::sort(m_params.begin(), m_params.end());
    m_params.erase(std::unique(m_params.begin(), m_params.end()),
            m_params.end());
    m_loss = op;
}
gradient_descent::~gradient_descent(){}

void gradient_descent::minibatch_learning(const unsigned int n_max_epochs, unsigned long int n_max_secs, bool randomize){
    unsigned int n_batches = current_batch_num();
    if(m_update_every==0)
        m_update_every = n_batches;
    std::vector<unsigned int> batchids;
    {   // prepare batch id vector
        batchids.resize(n_batches);
        for(unsigned int i=0;i<n_batches;i++)
            batchids[i] = i;
    }
    log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("gd"));
    unsigned long int iter = 1;
    unsigned long int wups = 0;
    try{
        m_stop_reason = SR_UNKNOWN;
        unsigned long int t_start = time(NULL);
        for (; ; ++m_epoch) {
            //log4cxx::MDC epoch_mdc("epoch",boost::lexical_cast<std::string>(m_epoch));
            TRACE1(log, "minibatch_learning_epoch", "epoch", boost::lexical_cast<std::string>(m_epoch));

            // stop if time limit is exceeded
            if(time(NULL) - t_start > n_max_secs) {
                LOG4CXX_WARN(log, "STOP minibatch learning: Timeout ("<<(time(NULL)-t_start)<<"s)");
                throw timeout_stop();
            }
            // stop if epoch limit is exceeded
            if(m_epoch >= n_max_epochs){
                if(m_learnrate != 0.f)
                    LOG4CXX_WARN(log, "STOP minibatch learning: Max epochs");
                throw max_iter_stop();
            }

            if(randomize)
                std::random_shuffle(batchids.begin(),batchids.end());

            before_epoch(m_epoch, wups); // may run early stopping

            for (unsigned int  batch = 0; batch < n_batches; ++batch, ++iter) {

                before_batch(m_epoch, batchids[batch]); // should load data into inputs

                m_swipe.fprop();  // forward pass
                //std::cout << "free mem after fprop: " << cuv::getFreeDeviceMemory()/1024/1024 << std::endl;

                if(m_learnrate){
                    // this is not an evaluation pass, we're actually supposed to do work ;)

                    m_swipe.bprop(); // backward pass
                    after_batch(m_epoch, batchids[batch]); // should accumulate errors etc

                    if(iter % m_update_every == 0) {
                        before_weight_update(wups);
                        update_weights(); 
                        wups ++;
                        after_weight_update(wups);
                    }
                }
                else{
                    after_batch(m_epoch, batchids[batch]); // should accumulate errors etc
                }
            }
            after_epoch(m_epoch, wups); // should log error etc
        }
    }catch(arithmetic_error_stop){
        m_stop_reason = SR_NAN;
    }catch(no_improvement_stop){
        m_stop_reason = SR_NO_IMPROVEMENT;
    }catch(convergence_stop){
        m_stop_reason = SR_CONVERGENCE;
    }catch(max_iter_stop){
        m_stop_reason = SR_MAX_ITER;
    }catch(timeout_stop){
        m_stop_reason = SR_TIMEOUT;
    }catch(network_stop){
        m_stop_reason = SR_NETWORK;
    }catch(gradient_descent_stop){
        m_stop_reason = SR_UNKNOWN;
    }

    // Restore parameters.
    // - may also restore m_epoch
    load_best_params();    
    //m_epoch *= n_batches; // number of batch presentations

    done_learning();
}

void gradient_descent::batch_learning(const unsigned int n_epochs, unsigned long int n_max_secs){
    try{
        m_stop_reason = SR_UNKNOWN;
        unsigned long int t_start = time(NULL);
        for (; m_epoch < n_epochs; ++m_epoch) {
            if(time(NULL) - t_start > n_max_secs)
                throw timeout_stop();
            before_epoch(m_epoch, m_epoch); // wups==epoch
            m_swipe.fprop();
            m_swipe.bprop();
            after_batch(m_epoch, 0); // should accumulate errors etc
            before_weight_update(m_epoch);
            update_weights();
            after_weight_update(m_epoch);
            after_epoch(m_epoch, m_epoch); // wups==epoch
            //m_learnrate *= m_learnrate_decay;
        }
    }catch(arithmetic_error_stop){
        m_stop_reason = SR_NAN;
    }catch(no_improvement_stop){
        m_stop_reason = SR_NO_IMPROVEMENT;
    }catch(convergence_stop){
        m_stop_reason = SR_CONVERGENCE;
    }catch(max_iter_stop){
        m_stop_reason = SR_MAX_ITER;
    }catch(timeout_stop){
        m_stop_reason = SR_TIMEOUT;
    }catch(network_stop){
        m_stop_reason = SR_NETWORK;
    }catch(gradient_descent_stop){
        m_stop_reason = SR_UNKNOWN;
    }
    load_best_params();
    done_learning();
}

void gradient_descent::update_weights(){
    for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++){
        //cuvAssert(&((*it)->result()->delta.cdata()));
        //cuvAssert(NULL != dynamic_cast<ParameterInput*>(*it));
        ParameterInput* inp = (ParameterInput*) *it;
        if(! inp->derivable()) continue;

        float lr = m_learnrate * inp->get_learnrate_factor();
        float wd = m_weightdecay * inp->get_weight_decay_factor();
        // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
        //       we're changing the underlying object all cow_ptrs pointing to it!!!
        cuv::learn_step_weight_decay( *inp->data_ptr().ptr(), inp->delta(), lr, wd);
        inp->reset_delta();
        //m_learnrate *= m_learnrate_decay;
    }
}

void gradient_descent::save_current_params(){
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end(); it++){
        ParameterInput* p = dynamic_cast<ParameterInput*>(*it);
        cuvAssert(p);
        m_best_perf_params[*it] = p->data();
    }
    m_epoch_of_saved_params = m_epoch;
}

void gradient_descent::forget_best_params(){
    m_best_perf_params.clear();
    m_epoch_of_saved_params = 0;
}

void gradient_descent::load_best_params(){
    log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("gd"));
    // load the best parameters again
    bool did_load_something = false;
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end(); it++){
        ParameterInput* p = dynamic_cast<ParameterInput*>(*it);
        cuvAssert(p);
        std::map<Op*,cuv::tensor<float, cuv::host_memory_space> >::iterator mit = m_best_perf_params.find(*it);
        if(mit != m_best_perf_params.end()) {
            p->data() = m_best_perf_params[*it];
            did_load_something = true;
        }
    }
    if(did_load_something)
    {
        LOG4CXX_INFO(log, "...loaded best params (epoch "<< m_epoch_of_saved_params <<")");
        m_epoch = m_epoch_of_saved_params;
    }
}

void gradient_descent::eval_epoch(unsigned int current_epoch){
    if(current_batch_num) {
        unsigned int n_batches = current_batch_num();
        for (unsigned int  batch = 0; batch < n_batches; ++batch) {
            before_batch(current_epoch, batch);
            m_swipe.fprop(); // fprop /only/
            after_batch(current_epoch, batch);
        }
    }else{
        // batch learning
        before_batch(current_epoch, 0);
        m_swipe.fprop(); // fprop /only/
        after_batch(current_epoch, 0);
    }
}

rprop_gradient_descent::rprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay)
    :gradient_descent(op, result, params, learnrate, weightdecay), m_learnrates(params.size()), m_old_dw(params.size()), m_l1decay(0.f)
{ 
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
        m_learnrates[i].resize(((ParameterInput*)*it)->data().shape());
        m_learnrates[i] = learnrate;

        m_old_dw[i].resize(((ParameterInput*)*it)->data().shape());
        m_old_dw[i] = (signed char)0;
    }

    after_batch.connect(boost::bind(&rprop_gradient_descent::inc_n_batches, this)); 
}

void rprop_gradient_descent::update_weights()
{
    using namespace cuv;
    unsigned int i=0;
    //cuvAssert(m_n_batches > 0);
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
        ParameterInput* param = dynamic_cast<ParameterInput*>(*it);
        if(! param->derivable()) continue;
        Op::value_type dW = param->delta(); 
        if(m_n_batches > 1)
            dW /= (float) m_n_batches;
        float wd = m_weightdecay * param->get_weight_decay_factor();
        //cuv::rprop(*param->data_ptr().ptr(), dW, m_old_dw[i], m_learnrates[i],  0.0000000f, m_weightdecay);
        cuv::rprop(*param->data_ptr().ptr(), dW, m_old_dw[i], m_learnrates[i], wd, m_l1decay);
        param->reset_delta();
    }
    m_n_batches = 0;
}


// ----------------------- spn gradient descent ------------------  

spn_gradient_descent::spn_gradient_descent(Op::op_ptr op, input_ptr X, input_ptr Y, unsigned int result, boost::shared_ptr<monitor> results, const paramvec_t& params, inf_type_ptr INFERENCE_TYPE, float learnrate, bool rescale_weights, float thresh, float weightdecay)
    :gradient_descent(op, result, params, learnrate, weightdecay), m_old_dw(params.size()), pt_X(X), pt_Y(Y), m_learnrate(learnrate), m_rescale(rescale_weights), m_l1decay(0.f), m_thresh(thresh)
{
        m_INFERENCE_TYPE = INFERENCE_TYPE;
        m_results = results;
        
        unsigned int nbatch = Y->data().shape(0);
        unsigned int nclass = Y->data().shape(1);
    //generate ops 
        value_ptr t0 ( new value_type(cuv::extents[nbatch][nclass]));
        labels = t0;

        value_ptr t1 ( new value_type(cuv::extents[nbatch]));
        S = t1;

        value_ptr t2 ( new value_type(cuv::extents[nbatch]));
        SM = t2;

        value_ptr t3 ( new value_type(cuv::extents[nbatch][nclass]));
        classification = t3;

        value_ptr t4 ( new value_type(cuv::extents[nbatch][nclass]));
        Y_oneOutOfN = t4;

        int_value_ptr t5 ( new int_value_type(cuv::extents[nbatch]));
        a1 = t5;
        
        int_value_ptr t6 ( new int_value_type(cuv::extents[nbatch]));
        a2 = t6;
        
        unsigned int i=0;
        for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
        m_old_dw[i].resize(((ParameterInput*)*it)->data().shape());
        m_old_dw[i] = (float)0;
    }

    after_batch.connect(boost::bind(&spn_gradient_descent::inc_n_batches, this));
    }


void spn_gradient_descent::minibatch_learning(const unsigned int n_max_epochs, unsigned long int n_max_secs, bool randomize){
    unsigned int n_batches = n_batch;
    
    if(m_update_every==0)
        m_update_every = n_batches;
    std::vector<unsigned int> batchids;
    {   // prepare batch id vector
        batchids.resize(n_batches);
        for(unsigned int i=0;i<n_batches;i++)
            batchids[i] = i;
    }
    

    log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("spn_gd"));        
    
    unsigned long int iter = 1;
    unsigned long int wups = 0;
    try{
        unsigned long int t_start = time(NULL);
        for (; ; ++m_epoch) {
            float c_err = 0;
            float s_err = 0;
            
            log4cxx::MDC epoch_mdc("epoch",boost::lexical_cast<std::string>(m_epoch));         

            // stop if time limit is exceeded
            if(time(NULL) - t_start > n_max_secs) {
                LOG4CXX_WARN(log, "STOP minibatch learning: Timeout ("<<(time(NULL)-t_start)<<"s)");
                throw timeout_stop();
            }
            // stop if epoch limit is exceeded
            if(m_epoch >= n_max_epochs){
                if(m_learnrate != 0.f)
                    LOG4CXX_WARN(log, "STOP minibatch learning: Max epochs");
                throw max_iter_stop();
            }

            if(randomize)
                std::random_shuffle(batchids.begin(),batchids.end());

            before_epoch(m_epoch, wups); // may run early stopping
            for (unsigned int  batch = 0; batch < n_batches; ++batch, ++iter) {

                before_batch(m_epoch, batchids[batch]);
                //TODO use signal
                get_batch(m_epoch, batchids[batch]);

                //  save labels;
                *labels = pt_Y->data().copy();
                
                //marginalize labels
                fill(pt_Y->data(), -1.f); // set labels to saved result..

                
                //marginalization run
                m_swipe.fprop();  // forward pass
                *SM = (*m_results)["S"].copy();
                m_swipe.bprop(); // backward pass
                
                *classification = pt_Y->delta().copy();
                cuv::apply_scalar_functor(*classification, cuv::SF_EXP);                        
                cuv::reduce_to_col(*a1, *classification, cuv::RF_ARGMAX);
                cuv::reduce_to_col(*a2, *Y_oneOutOfN, cuv::RF_ARGMAX);
                    
                *a1 -= *a2;
                int n_wrong = batch_size - cuv::count(*a1,0);
                float tmp_cerr = n_wrong / (float) batch_size;
                c_err += tmp_cerr;

                bool log_batch_stats = false;        
	if (log_batch_stats){
                if(m_learnrate){
                    // this is not an evaluation pass, we're actually supposed to do work ;)
                    log4cxx::MDC batch_err_class_mdc("batch_err_class",boost::lexical_cast<std::string>(tmp_cerr));    
                    LOG4CXX_WARN(log, "logging batch stats ("<<(time(NULL)-t_start)<<"s)");
		} else{	
                    log4cxx::MDC batch_err_spn_mdc("batch_eval_class",boost::lexical_cast<std::string>(tmp_cerr)); 
                    LOG4CXX_WARN(log, "logging eval batch stats ("<<(time(NULL)-t_start)<<"s)");
		}   
	} 
         	//save old derivatives
                    unsigned int i=0;
                    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
                        ParameterInput* param = dynamic_cast<ParameterInput*>(*it);
                        if(! param->derivable()) continue;
                        Op::value_type dW = param->delta();

                        //save old result
                        m_old_dw[i] = dW.copy();
                        param->reset_delta();
                    }
                    
                    pt_Y->data() = *labels;
                    m_swipe.fprop();
                    *S = (*m_results)["S"].copy();                        

                    m_swipe.bprop();
                    
                    cuv::apply_scalar_functor(*SM, cuv::SF_EXP);
                    cuv::apply_scalar_functor(*S, cuv::SF_EXP);
                    
//                        std::cout << std::endl << "S: " << cuv::mean(*S) << std::endl;
//                        std::cout << "SM: " << cuv::mean(*SM) << std::endl;
                    
                    cuv::apply_binary_functor(*SM, *S, cuv::BF_SUBTRACT);
                    cuv::apply_scalar_functor(*SM, *SM, cuv::SF_ABS);                             
                    float tmp_err = cuv::mean(*SM);
                    s_err += tmp_err;                        

                    // nan check
                    if ((s_err != s_err) || (c_err != c_err)) throw std::runtime_error("NAN occured -> Abort");
                    
                    after_batch(m_epoch, batchids[batch]); // should accumulate errors etc
                    
                    
  //              if(m_learnrate){
                    if (log_batch_stats){
		        log4cxx::MDC batch_err_spn_mdc("batch_err_spn",boost::lexical_cast<std::string>(tmp_err)); 
                        LOG4CXX_WARN(log, "logging batch stats ("<<(time(NULL)-t_start)<<"s)");
		    }
                    if(iter % m_update_every == 0) {
                        //std::cout << std::endl;
                        before_weight_update(wups);
                        update_weights(); 
                        wups ++;
                        after_weight_update(wups);
                    }
//                } else{
                   // after_batch(m_epoch, batchids[batch]); // should accumulate errors etc           
//                }
            }
            //logging
 
            if (!m_learnrate){
		float eval_error = c_err / float(n_batches);
                log4cxx::MDC eval_batch_mdc("eval_class",boost::lexical_cast<std::string>(eval_error)); 
                std::cout << " eval: classification err: "  << eval_error << std::endl;    
                LOG4CXX_WARN(log, "logging eval ("<<(time(NULL)-t_start)<<"s)");

/*                    std::string name = "class_";
                    name.append(std::to_string(batchids[batch]));
                    name.append("_");
                    name.append(std::to_string(m_epoch));

                    tofile(name, *classification);
                    tofile("labels", *Y_oneOutOfN);
  */                      

            } else {
		float eval_error = c_err / float(n_batches);
		m_spn_err = s_err / float(n_batches);
		m_class_err = eval_error;
                log4cxx::MDC err_spn_mdc("err_spn",boost::lexical_cast<std::string>(m_spn_err)); 
                log4cxx::MDC err_class_mdc("err_class",boost::lexical_cast<std::string>(eval_error)); 
                LOG4CXX_WARN(log, "logging  errors"<<(time(NULL)-t_start)<<"s)");
  
                std::cout << "spn err: "  << m_spn_err;
                std::cout << ", classification err: "  << eval_error << std::endl;                     
        }
        m_results->log_to_file();            
        after_epoch(m_epoch, wups); // should log error etc
        }

    }catch(gradient_descent_stop){
    }

    // Restore parameters.
    // - may also restore m_epoch
    load_best_params();    
    //m_epoch *= n_batches; // number of batch presentations

    done_learning();
}



void spn_gradient_descent::update_weights()
{
    using namespace cuv;
    //do gradient descent for every parameter
    unsigned int i = 0;
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
        ParameterInput* param = dynamic_cast<ParameterInput*>(*it);
        if(! param->derivable()) continue;
        Op::value_type dW = param->delta();
        if(m_n_batches > 1)
        {
            dW /= (float) m_n_batches;
            m_old_dw[i] /= (float) m_n_batches;
        }
/*
	std::string S("fS_");
	std::string SM("fSM_");
        SM.append(std::to_string(i));
        S.append(std::to_string(i));

	tofile(SM, m_old_dw[i]) ;
	tofile(S, dW);
*/
        float wd = m_weightdecay * param->get_weight_decay_factor();
        float rate = m_learnrate * param->get_learnrate_factor();
        bool hard_inf = m_INFERENCE_TYPE->at(i);
        spn_gd(*param->data_ptr().ptr(), m_old_dw[i],  dW, hard_inf, m_rescale, m_thresh, rate, wd, m_l1decay);
        param->reset_delta();
    }
    m_n_batches = 0;
}


// ------------ momentum gradient descent  ---------  \\-
momentum_gradient_descent::momentum_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay, float momentum)
    :gradient_descent(op, result, params, learnrate, weightdecay), m_last_delta(params.size()), m_momentum(momentum)
{ 
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
        m_last_delta[i].resize(((ParameterInput*)*it)->data().shape());
        m_last_delta[i] = 0.f;
    }
}

void momentum_gradient_descent::update_weights(){
    using namespace cuv;
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
        ParameterInput* inp = (ParameterInput*) *it;
        if(! inp->derivable()) continue;

        float lr = m_learnrate * inp->get_learnrate_factor();
        float wd = m_weightdecay * inp->get_weight_decay_factor();

        cuvAssert(inp->delta().shape() == inp->data().shape());
#if 1

        // this is the momentum part:
        cuv::apply_binary_functor(m_last_delta[i], inp->delta(), cuv::BF_AXPY, m_momentum);

        // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
        //       we're changing the underlying object all cow_ptrs pointing to it!!!
        cuv::learn_step_weight_decay( *inp->data_ptr().ptr(), m_last_delta[i], lr, wd);
        //m_learnrate *= m_learnrate_decay;
#else
        // when gradient is exactly zero, assume that the unit has been dropped out
        // then, do not apply any gradient step, not even the part due to momentum
        cuv::tensor<unsigned char, matrix::memory_space_type> mask =
            inp->delta() == 0.f;

        matrix last_delta_copy = m_last_delta[i].copy();
        // This is the momentum part for imagenet:
        //cuv::apply_binary_functor(m_last_delta[i], inp->delta(), cuv::BF_AXPY, m_momentum);
        // this is the momentum part for mnist:
        cuv::apply_binary_functor(m_last_delta[i], inp->delta(), cuv::BF_AXPBY, m_momentum, (-1.f + m_momentum)*lr);  // lastDelta += momentum * newDelta


        // copy back the part where inp->delta() was zero
        cuv::apply_scalar_functor(m_last_delta[i], last_delta_copy, cuv::SF_COPY, &mask);  // lastDelta[newDelta==0] = originalLastDelta


        // ensure that only weights are changed which had non-zero inp->delta().
        last_delta_copy = m_last_delta[i].copy();
        cuv::apply_scalar_functor(last_delta_copy, cuv::SF_MULT, 0.f, &mask);
        cuvAssert(wd == 0.f); // if wd != 0, the learn_step_weight_decay
        // function would still update the zero gradient weights!



        // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
        //       we're changing the underlying object all cow_ptrs pointing to it!!!
        // cuv::learn_step_weight_decay( *inp->data_ptr().ptr(), last_delta_copy, lr, wd);
        *inp->data_ptr().ptr()  += last_delta_copy;
#endif
        inp->reset_delta();
    }
}

// ------------ rmsprop gradient descent  ---------  \\-
rmsprop_gradient_descent::rmsprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay, float delta, float grad_avg, float l1penalty)
    :gradient_descent(op, result, params, learnrate, weightdecay),
    m_sq_grad_sum(params.size()),
    m_delta(delta),
    m_grad_avg_const(grad_avg),
    m_l1penalty(l1penalty)
{ 
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
        m_sq_grad_sum[i].resize(((ParameterInput*)*it)->data().shape());
        m_sq_grad_sum[i] = 1.f;
    }
}

void rmsprop_gradient_descent::update_weights(){
    using namespace cuv;
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
        ParameterInput* inp = (ParameterInput*) *it;
        if(! inp->derivable()) continue;

        // we exploit CUVs problems with const-correctness to get an
        // overwritable version of inp->delta.
        // the delta() of const ParameterInput* uses cdata() of the
        // cow_ptr!
        matrix delta = ((const ParameterInput*) inp)->delta();

        // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
        //       we're changing the underlying object all cow_ptrs pointing to it!!!
        cuv::libs::opt::rmsprop(*inp->data_ptr().ptr(),delta,m_sq_grad_sum[i],m_learnrate * inp->get_learnrate_factor(),m_delta,m_weightdecay * inp->get_weight_decay_factor(),m_l1penalty,m_grad_avg_const);

        inp->reset_delta();
    }
}
// ------------ adagrad gradient descent  ---------  \\-
adagrad_gradient_descent::adagrad_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay, float delta, int winsize, float l1penalty)
    :gradient_descent(op, result, params, learnrate, weightdecay),
    m_sq_grad_sum(params.size()),
    m_delta(delta),
    m_winsize(winsize),
    m_count(0),
    m_l1penalty(l1penalty)
{ 
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
        m_sq_grad_sum[i].resize(((ParameterInput*)*it)->data().shape());
        m_sq_grad_sum[i] = 0.f;
    }
}

void adagrad_gradient_descent::update_weights(){
    using namespace cuv;
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
        ParameterInput* inp = (ParameterInput*) *it;
        if(! inp->derivable()) continue;

        // we exploit CUVs problems with const-correctness to get an
        // overwritable version of inp->delta.
        // the delta() of const ParameterInput* uses cdata() of the
        // cow_ptr!
        matrix delta = ((const ParameterInput*) inp)->delta();
        float lr = m_learnrate * inp->get_learnrate_factor();
        float wd = m_weightdecay * inp->get_weight_decay_factor();
        float wd1 = m_l1penalty * inp->get_weight_decay_factor();

        // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
        //       we're changing the underlying object all cow_ptrs pointing to it!!!
        cuv::libs::opt::adagrad(*inp->data_ptr().ptr(),delta,m_sq_grad_sum[i],lr,m_delta,wd,wd1);

        if(m_count % m_winsize == 0)
        {
            m_sq_grad_sum[i] = 0.f;
        }

        inp->reset_delta();
    }
    ++m_count;
}
// ------------ accelerated gradient descent  ---------  \\-
accelerated_gradient_descent::accelerated_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay, float p)
    :gradient_descent(op, result, params, learnrate, weightdecay),
    m_w_ag(params.size()),
    m_beta(1),
    m_count(0),
    m_p(p)
{ 
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
        m_w_ag[i] = ((ParameterInput*)*it)->data().copy(); 
    }
}

void accelerated_gradient_descent::finish(){
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++, i++){
        ((ParameterInput*)*it)->data() = m_w_ag[i];
    }
}
void accelerated_gradient_descent::update_weights(){
    using namespace cuv;
    unsigned int i=0;
    for(paramvec_t::iterator it=m_params.begin(); it!=m_params.end();it++, i++){
        ParameterInput* inp = (ParameterInput*) *it;
        if(! inp->derivable()) continue;

        // we exploit CUVs problems with const-correctness to get an
        // overwritable version of inp->delta.
        // the delta() of const ParameterInput* uses cdata() of the
        // cow_ptr!
        matrix delta = ((const ParameterInput*) inp)->delta();

        // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
        //       we're changing the underlying object all cow_ptrs pointing to it!!!

        float wd = m_weightdecay * inp->get_weight_decay_factor();
        float gamma_i = m_learnrate * inp->get_learnrate_factor() * std::pow(m_count, m_p);
        float beta_i_inv  = 1.f / ((m_count+1.f)/2.f);

        // in the paper, the loss is evaluated on w^{md}, which is a mix of w and w^{ag}.
        // We cannot do this here, since the loss has already been calculated... we have to
        // change the order. Our inp holds the w^md, and (this is what's
        // "wrong" here) we return w^md, not w^ag. Call finish() to get the w^ag weights.
        matrix& wmd    = *inp->data_ptr().ptr();
        cuv::learn_step_weight_decay(wmd,delta,gamma_i,wd);

        // calculate new w^{ag}
        cuv::apply_binary_functor(m_w_ag[i],wmd,m_w_ag[i],cuv::BF_AXPBY,beta_i_inv, 1.f-beta_i_inv);

        // calculate new w
        cuv::apply_binary_functor(wmd,wmd,m_w_ag[i],cuv::BF_AXPBY,beta_i_inv, 1.f-beta_i_inv);

        inp->reset_delta();
        //if(m_count % 100 == 0)
        //    the norm goes towards zero as beta increases
        //    std::cout << "beta_i_inv: "<<beta_i_inv<<" cuv::norm2(wmd-m_w_ag[i]):" << cuv::norm2(wmd-m_w_ag[i]) << std::endl;
    }
    ++m_count;
}


/*********************************************
    * convergence checking 
    *********************************************/
convergence_checker::convergence_checker(
        gradient_descent& gd,
        boost::function<float(void)> performance,
        float thresh, unsigned int min_wups, float patience_inc_fact)
    :   m_gd(gd),
        m_performance(performance),
        m_thresh(thresh),
        m_patience(min_wups),
        m_patience_inc_fact(patience_inc_fact),
        m_max_steps(1),
        m_steps(0),
        m_lr_fact(1.f)
{
    cuvAssert(patience_inc_fact > 1.);
    m_connection = gd.after_epoch.connect(boost::ref(*this), boost::signals::at_front);
}


void convergence_checker::disconnect(){
    m_connection.disconnect();
    assert(!m_connection.connected());
}

void convergence_checker::operator()(unsigned int current_epoch, unsigned int wups){
    log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("conv_check"));
    float perf = m_performance();
    if(perf != perf){
        LOG4CXX_WARN( log,  "STOP got NaN in convergence check!");
        throw arithmetic_error_stop();
    }
    if(current_epoch == 0){
        m_last_perf = perf;
        return;
    }
    LOG4CXX_DEBUG( log, current_epoch<<": "<<wups<<"/"<<m_patience<<", "<<std::setprecision(3) <<(perf/m_last_perf)<<": "<<std::setprecision(6)<<perf );
    if(perf < m_thresh * m_last_perf){
        m_last_perf = perf;
        m_patience = std::max(m_patience, (unsigned int)(m_patience_inc_fact*wups));

        //m_gd.save_current_params(); // consider everything that did not go below threshold as "overtraining".
    }

    if(wups >= m_patience){
        m_steps ++;
        if(m_steps >= m_max_steps){
            LOG4CXX_WARN( log,"STOP after "<<current_epoch<<" epochs:"<< perf);
            throw convergence_stop();
        }
        m_gd.set_learnrate(m_gd.learnrate() * m_lr_fact);
        LOG4CXX_WARN(log, "converged: decreasing learnrate");
        m_patience = std::max(m_patience, (unsigned int)(m_patience_inc_fact*wups));
    }
    else if( perf > 1.1f * m_last_perf ){
        //m_gd.decay_learnrate(m_lr_fact);
        //LOG4CXX_WARN(log, "unsteady: decreasing learnrate");
    }
}

void convergence_checker::decrease_lr(unsigned int max_steps, float lr_fact){
    m_max_steps = max_steps;
    m_lr_fact   = lr_fact;
}

/*********************************************
    * early stopping
    *********************************************/
early_stopper::early_stopper(
        gradient_descent& gd,
        boost::function<float(void)> performance,
        float thresh, unsigned int every, float patience_increase, unsigned int box_filter_size)
    : m_gd(gd)
    , m_performance(performance)
    , m_every(every)
    , m_thresh(thresh)
    , m_patience_increase(patience_increase)
    , m_box_filter_size(box_filter_size)
{
    cuvAssert(patience_increase > 1.);
    m_best_perf = std::numeric_limits<float>::infinity();
    m_connection = gd.before_epoch.connect(boost::ref(*this), boost::signals::at_front);
    m_patience = 4000;
}

void early_stopper::disconnect(){
    m_connection.disconnect();
    assert(!m_connection.connected());
}

void early_stopper::operator()(unsigned int current_epoch, unsigned int wups){
    log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("early_stop"));
    log4cxx::NDC ndc("early_stopper");
    if(current_epoch%m_every!=0)
        return;

    // this does the actual work, it runs fprop on all ES batches
    before_early_stopping_epoch(current_epoch);
    m_gd.eval_epoch(current_epoch);

    // determine how good we've been (usually set to some perf()
    // function of the construct you're trying to minimize)
    float perf = m_performance();

    // call after_early_stopping_epoch() HERE, so that
    // m_performance can rely on still being in validation mode!
    after_early_stopping_epoch(current_epoch);

    if(perf != perf){
        LOG4CXX_WARN( log,  "STOP Got NaN in early-stopping check!");
        throw arithmetic_error_stop();
    }

    m_val_perfs.push_back(perf);

    perf = 0.f;
    unsigned int window_size = std::min(m_val_perfs.size(), (size_t) m_box_filter_size);
    for(unsigned int i = m_val_perfs.size()-window_size;
            i < m_val_perfs.size(); 
            i++)
        perf += m_val_perfs[i];
    perf /= window_size;


    log4cxx::MDC perf_mdc("perf", boost::lexical_cast<std::string>(perf));
    log4cxx::MDC patience_mdc("patience", boost::lexical_cast<std::string>(m_patience));
    log4cxx::MDC wups_mdc("wups", boost::lexical_cast<std::string>(wups));
    if(perf < m_best_perf) {
        log4cxx::MDC perf_mdc("best_perf", boost::lexical_cast<std::string>(perf));
        LOG4CXX_DEBUG(log, "* "<<current_epoch<<": "<<wups<<" / "<<m_patience<<", "<<std::setprecision(3) <<(perf/m_best_perf)<<": "<< std::setprecision(6) << perf);
    } else {
        log4cxx::MDC perf_mdc("best_perf", boost::lexical_cast<std::string>(m_best_perf));
        LOG4CXX_DEBUG(log, "- "<<current_epoch<<": "<<wups<<" / "<<m_patience<<", "<<std::setprecision(3) <<(perf/m_best_perf)<<": "<< std::setprecision(6) << perf);
    }

    if(perf < m_best_perf) {
        // save the (now best) parameters
        m_gd.save_current_params();  
        improved();
        if(perf < m_thresh * m_best_perf){ 
            // improved by more than thresh
            m_patience = std::max((float)m_patience, (float)wups * m_patience_increase);
        }
        m_best_perf = perf;
    }

    if(m_best_perf == 0.f){
        log4cxx::MDC mdc("best_perf", boost::lexical_cast<std::string>(m_best_perf));
        LOG4CXX_WARN(log, "STOP loss zero after " <<current_epoch << " epochs");
        throw no_improvement_stop();
    }
    if(m_patience <= wups){
        // stop learning if we failed to get significant improvements
        log4cxx::MDC mdc("best_perf", boost::lexical_cast<std::string>(m_best_perf));
        LOG4CXX_WARN(log, "STOP no improvement after " <<current_epoch << " epochs");
        throw no_improvement_stop();
    }

}

}
