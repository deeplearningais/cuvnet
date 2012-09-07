#include<boost/limits.hpp>
#include<cuv/tensor_ops/rprop.hpp>
#include "gradient_descent.hpp"
#include <cuv/tools/device_tools.hpp>

namespace cuvnet
{
    gradient_descent::gradient_descent(const Op::op_ptr& op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay)
        : m_result(result), m_params(params), m_learnrate(learnrate), m_learnrate_decay(1.f), m_weightdecay(weightdecay)
          , m_epoch(0), m_epoch_of_saved_params(0)
          , m_swipe(*op,result,params), m_convergence_checking(false)
          , m_patience(4)
          , m_best_perf(std::numeric_limits<float>::infinity())
    { 
        m_loss = op;
    }
    gradient_descent::~gradient_descent(){}

    void gradient_descent::minibatch_learning(const unsigned int n_max_epochs, unsigned long int n_max_secs, unsigned int update_every, bool randomize){
        unsigned int n_batches = current_batch_num();
        if(update_every==0)
            update_every = n_batches;
        std::vector<unsigned int> batchids;
        {   // prepare batch id vector
            batchids.resize(n_batches);
            for(unsigned int i=0;i<n_batches;i++)
                batchids[i] = i;
        }
        unsigned long int iter = 1;
        try{
            unsigned long int t_start = time(NULL);
            for (m_epoch = 0; ; ++m_epoch) {

                // stop if time limit is exceeded
                if(time(NULL) - t_start > n_max_secs) {
                    std::cout << "Minibatch Learning Timeout ("<<(time(NULL)-t_start)<<"s)" << std::endl;/* cursor */
                    throw timeout_stop();
                }
                // stop if epoch limit is exceeded
                if(iter/n_batches >= n_max_epochs){

                    throw max_iter_stop();
                }

                if(randomize)
                    std::random_shuffle(batchids.begin(),batchids.end());

                before_epoch(m_epoch); // may run early stopping

                for (unsigned int  batch = 0; batch < n_batches; ++batch, ++iter) {

                    before_batch(m_epoch, batchids[batch]); // should load data into inputs

                    m_swipe.fprop();  // forward pass
                    //std::cout << "free mem after fprop: " << cuv::getFreeDeviceMemory()/1024/1024 << std::endl;

                    if(m_learnrate && !(m_convergence_checking && m_epoch==0)){
                        // this is not an evaluation pass, we're actually supposed to do work ;)

                        m_swipe.bprop(); // backward pass

                        if(iter % update_every == 0)
                            update_weights(); 
                    }
                    after_batch(m_epoch, batchids[batch]); // should accumulate errors etc
                }
                after_epoch(m_epoch); // should log error etc

            }
        }catch(timeout_stop){
        }catch(no_improvement_stop){
        }catch(convergence_stop){
        }catch(max_iter_stop){
        }catch(network_stop){
            std::cout << "stopping since others in network stopped" << std::endl;
        }

        // Restore parameters.
        // - may also restore m_epoch
        load_best_params();    
        //m_epoch *= n_batches; // number of batch presentations

        done_learning();
    }

    void gradient_descent::batch_learning(const unsigned int n_epochs, unsigned long int n_max_secs){
                unsigned long int t_start = time(NULL);
                for (unsigned int epoch = 0; epoch < n_epochs; ++epoch) {
                    if(time(NULL) - t_start > n_max_secs)
                        break;
                    before_epoch(epoch);
                    m_swipe.fprop();
                    m_swipe.bprop();
                    after_batch(epoch, 0); // should accumulate errors etc
                    update_weights();
                    after_epoch(epoch);
                    m_learnrate *= m_learnrate_decay;
                }
                done_learning();
    }

    void gradient_descent::update_weights(){
        for(paramvec_t::iterator it=m_params.begin();it!=m_params.end();it++){
            //cuvAssert(&((*it)->result()->delta.cdata()));
            //cuvAssert(NULL != dynamic_cast<ParameterInput*>(*it));
            ParameterInput* inp = (ParameterInput*) *it;

            float lr = m_learnrate * inp->get_learnrate_factor();
            float wd = m_weightdecay * inp->get_weight_decay_factor();
            // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
            //       we're changing the underlying object all cow_ptrs pointing to it!!!
            cuv::learn_step_weight_decay( *inp->data_ptr().ptr(), inp->delta(), -lr, wd);
            inp->reset_delta();
            m_learnrate *= m_learnrate_decay;
        }
    }

    void gradient_descent::early_stop_test(unsigned int every, float thresh, float patience_increase, unsigned int current_epoch, unsigned int box_filter_size){
        if(current_epoch%every!=0)
            return;

        // this does the actual work, it runs fprop on all ES batches
        before_early_stopping_epoch(current_epoch);
        early_stopping_epoch(current_epoch);

        // determine how good we've been (usually set to some perf()
        // function of the construct you're trying to minimize)
        float perf = m_performance();

        // call after_early_stopping_epoch() HERE, so that
        // m_performance can rely on still being in validation mode!
        after_early_stopping_epoch(current_epoch);

        m_val_perfs.push_back(perf);

        perf = 0.f;
        unsigned int window_size = std::min(m_val_perfs.size(), (size_t) box_filter_size);
        for(unsigned int i = m_val_perfs.size()-window_size;
                i < m_val_perfs.size(); 
                i++)
            perf += m_val_perfs[i];
        perf /= window_size;

        if(current_epoch == 0)
            m_initial_performance = perf;

        if(perf < m_best_perf)
            std::cout << "\r * early-stopping(epoch "<<current_epoch<<" / "<<m_patience<<", "<<(perf/m_best_perf)<<"): "<< perf<<std::flush;
        else
            std::cout << "\r - early-stopping(epoch "<<current_epoch<<" / "<<m_patience<<", "<<(perf/m_best_perf)<<"): "<< perf<<std::flush;

        if(perf < m_best_perf) {
            // save the (now best) parameters
            save_current_params();  
            if(perf < thresh * m_best_perf){ 
                // improved by more than thresh
                m_patience = std::max((float)m_patience, (float)current_epoch * patience_increase);
            }
            m_best_perf = perf;
        }

        if(m_patience <= current_epoch){
            // stop learning if we failed to get significant improvements
            load_best_params();
            throw no_improvement_stop();
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

    void gradient_descent::load_best_params(){
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
            std::cout << "...loaded best params (epoch "<< m_epoch_of_saved_params <<")"<<std::endl;
            m_epoch = m_epoch_of_saved_params;
        }
    }

    void gradient_descent::convergence_test(float thresh, unsigned int min_epochs, float patience_inc_factor, unsigned int current_epoch){
        float perf = m_performance();
        if(current_epoch == 0){
            m_initial_performance = perf;
            m_last_perf = perf;
            m_convcheck_patience = min_epochs;
            return;
        }
        std::cout << "\r * convergence-test("<<current_epoch<<"/"<<m_convcheck_patience<<", "<<(perf/m_last_perf)<<"): "<<perf<<"                  " << std::flush;
        if(perf < thresh * m_last_perf){
            m_last_perf = perf;
            m_convcheck_patience = std::max(m_convcheck_patience, (unsigned int)(patience_inc_factor*current_epoch));
        }

        if(current_epoch >= m_convcheck_patience){
            std::cout << "\r - convergence-test(stopping after "<<current_epoch<<" epochs):"<< perf << "                "<< std::endl;
            throw convergence_stop();
        }
    }

    unsigned int gradient_descent::early_stopping_epoch(unsigned int current_epoch){
        unsigned int n_batches = current_batch_num();
        for (unsigned int  batch = 0; batch < n_batches; ++batch) {
            before_batch(current_epoch, batch);
            m_swipe.fprop(); // fprop /only/
            after_batch(current_epoch, batch);
        }
        return n_batches;
    }

    rprop_gradient_descent::rprop_gradient_descent(Op::op_ptr op, unsigned int result, const paramvec_t& params, float learnrate, float weightdecay)
        :gradient_descent(op, result, params, learnrate, weightdecay), m_learnrates(params.size()), m_old_dw(params.size())
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
            Op::value_type dW = ::operator-(param->delta()); // TODO: change sign in cuv::rprop
            if(m_n_batches > 1)
                dW /= (float) m_n_batches;
            float wd = m_weightdecay * param->get_weight_decay_factor();
            //cuv::rprop(*param->data_ptr().ptr(), dW, m_old_dw[i], m_learnrates[i],  0.0000000f, m_weightdecay);
            cuv::rprop(*param->data_ptr().ptr(), dW, m_old_dw[i], m_learnrates[i], wd, 0.0000000f);
            param->reset_delta();
        }
        m_n_batches = 0;
    }

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

            float lr = m_learnrate * inp->get_learnrate_factor();
            float wd = m_weightdecay * inp->get_weight_decay_factor();

            cuvAssert(inp->delta().shape() == inp->data().shape());
            //std::cout << "cuv::norm1(m_last_delta[i]) " <<inp->name()<<" "<< cuv::norm1(m_last_delta[i])/m_last_delta[i].size() << std::endl;
            //std::cout << "      inp->delta()          " <<inp->name()<<" "<< cuv::minimum(inp->delta())<<" "<<cuv::mean(inp->delta())<<" "<<cuv::maximum(inp->delta()) << std::endl;

            // this is the momentum part:
            cuv::apply_binary_functor(m_last_delta[i], inp->delta(), cuv::BF_AXPY, m_momentum);

            // NOTE: inp->ptr() is accessing w/o the write-protection of the cow_ptr!!!!
            //       we're changing the underlying object all cow_ptrs pointing to it!!!
            cuv::learn_step_weight_decay( *inp->data_ptr().ptr(), m_last_delta[i], -lr, wd);
            m_learnrate *= m_learnrate_decay;
            inp->reset_delta();
        }
    }

}
