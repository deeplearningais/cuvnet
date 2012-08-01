#include <mongo/client/dbclient.h>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>                                                                  
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <tools/visualization.hpp>
#include <tools/gradient_descent.hpp>
#include <datasets/mnist.hpp>
#include <datasets/splitter.hpp>

#include <tools/preprocess.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <tools/monitor.hpp>

#include <mdbq/client.hpp>

using namespace mdbq; 
using namespace cuvnet;
namespace ll = boost::lambda;
typedef boost::shared_ptr<ParameterInput> input_ptr;
typedef boost::shared_ptr<Op> op_ptr;

/**
 * load a batch from the dataset
 */
void load_batch(
        boost::shared_ptr<ParameterInput> input,
        boost::shared_ptr<ParameterInput> target,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        cuv::tensor<float,cuv::dev_memory_space>* labels,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    input->data() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    target->data() = (*labels)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
}

struct hyperopt_client                                                                       
: public Client{                                                                             
    int id;                                                                                  
    boost::asio::io_service ios;                                                             
    monitor* m_mon;
    gradient_descent* m_gd;
    op_ptr m_loss;
    hyperopt_client(int i, std::string a, std::string b)                                     
        : Client(a,b), id(i){                                                                    
        }                                                                                        
    float test_phase(input_ptr in, input_ptr tch, dataset* ds, int bs){
        float mean = 0.f;
        {
            matrix data   = ds->val_data;
            matrix labels = ds->val_labels;
            std::vector<Op*> params; // empty!
            gradient_descent gd(m_loss, 0, params, 0);
            gd.register_monitor(*m_mon);
            gd.before_batch.connect(boost::bind(load_batch,in,tch, &data, &labels, bs, _2));
            gd.current_batch_num.connect(data.shape(0)/ll::constant(bs));
            m_mon->set_is_train_phase(false);
            gd.minibatch_learning(1, 100, 0);
            m_mon->set_is_train_phase(true);
            mean = m_mon->mean("classification error");
        }
        m_gd->repair_swiper();
        return mean;
    }
    float loss(float learnrate, float wd, int bs){
        // see logistic_regression.cpp for details
        mnist_dataset mnist("/home/local/datasets/MNIST");
        global_min_max_normalize<> n;
        n.fit_transform(mnist.train_data);
        n.transform(mnist.test_data);
        splitter splits(mnist, 1);
        dataset ds = splits[0];
        boost::shared_ptr<ParameterInput> input( new ParameterInput(cuv::extents[bs][ds.train_data.shape(1)],"input"));
        boost::shared_ptr<ParameterInput> target( new ParameterInput(cuv::extents[bs][ds.train_labels.shape(1)],"target"));
        logistic_regression lr(input, target);
        std::vector<Op*> params = lr.params();

        monitor mon(false);
        mon.add(monitor::WP_SCALAR_EPOCH_STATS, lr.get_loss(), "total loss");
        mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, lr.classification_error(), "classification error");

        matrix train_data = ds.train_data;
        matrix train_labels(ds.train_labels);

        gradient_descent gd(lr.get_loss(),0,params, learnrate, -wd);
        m_mon = &mon; m_gd = &gd; m_loss = lr.get_loss();
        gd.register_monitor(mon);
        gd.before_batch.connect(boost::bind(load_batch,input, target,&train_data, &train_labels, bs,_2));
        gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));
        gd.setup_early_stopping(boost::bind(&hyperopt_client::test_phase, this, input, target, &ds, bs), 5, 1.f, 2.f);
        gd.minibatch_learning(100000, 60*60); // 10 minutes maximum
        
        return gd.best_perf();
    }
    void handle_task(const mongo::BSONObj& o){                                               
        std::string cmd0 = o["cmd"].Array()[0].String();                                     
        std::string cmd1 = o["cmd"].Array()[1].String();                                     

        if(cmd0 != "bandit_json evaluate")                                                   
            throw std::runtime_error("HOC: do not understand cmd: `" + cmd0 +"'");           
        if(cmd1 != "sample_bandit.SampleBandit")                                            
            throw std::runtime_error("HOC: do not know bandit: `" + cmd1 +"'");              

        float lr = o["vals"]["lr"].Array()[0].Double();
        float wd = o["vals"]["wd"].Array()[0].Double();
        int   bs = std::pow(2.0,o["vals"]["bs"].Array()[0].Int());

        finish(BSON("status"<<"ok"<<"loss"<<loss(lr,wd,bs)));
    }
    void run(){
        cuv::initCUDA(id);
        cuv::initialize_mersenne_twister_seeds();
        this->reg(ios,1);
        ios.run();
    }
};

int
main(int argc, char **argv)
{
    // start n_clt worker threads
    static const int n_clt = 4;
    hyperopt_client* clients[n_clt];
    boost::thread*   threads[n_clt];

    for (int i = 0; i < n_clt; ++i) {
        clients[i] = new hyperopt_client(i, "131.220.7.92", "hyperopt");
        threads[i] = new boost::thread(boost::bind(&hyperopt_client::run, clients[i]));
    }

    for (int i = 0; i < n_clt; ++i) {
        threads[i]->join();
    }

    return 0;
}

