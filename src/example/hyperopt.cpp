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

/**
 * demonstrate how to use hyperopt to optimize the hyper-parameters for
 * logistic regression.
 */
struct hyperopt_client                                                                       
: public Client{                                                                             
    int id;                      ///< a unique ID for this client 
    boost::asio::io_service ios; ///< for serializing operations within client
    monitor* m_mon;              ///< monitor to keep track of learning progress
    gradient_descent* m_gd;      ///< gradient_descent object used for TRAINING
    op_ptr m_loss;               ///< the loss to be optimized
    unsigned int m_n_epochs;     ///< number of epochs until best VALIDATION attained
    dataset m_ds;    ///< the dataset containing ALL data
    matrix m_data;   ///< stores current inputs
    matrix m_labels; ///< stores current teachers
    /**
     * create a client.
     * @param i    a unique identifier of the client
     * @param host where database is hosted
     * @param db name of the database
     */
    hyperopt_client(int i, std::string host, std::string db)
        : Client(host, db, BSON("exp_key" << "sample_bandit.SampleBandit/hyperopt.tpe.TreeParzenEstimator"))
        , id(i), m_n_epochs(0){ }                                                                                        

    unsigned int n_batches(unsigned int bs){
        return m_data.shape(0)/bs;
    }
    void before_validation_epoch(monitor* mon){
        mon->set_training_phase(CM_VALID,0);
        m_data = m_ds.val_data;
        m_labels = m_ds.val_labels;
    }
    void after_validation_epoch(monitor* mon){
        mon->set_training_phase(CM_TRAIN,0);
        m_data = m_ds.train_data;
        m_labels = m_ds.train_labels;
    }

    /**
     * determine the loss for the given *hyper parameters*.
     *
     * @param learnrate the learnrate used in weight updates
     * @param wd weight decay strength
     * @param bs batch size
     */
    float loss(float learnrate, float wd, int bs){
        // see logistic_regression.cpp for details
        mnist_dataset mnist("/home/local/datasets/MNIST");
        global_min_max_normalize<> n;
        n.fit_transform(mnist.train_data);
        n.transform(mnist.test_data);
        splitter splits(mnist, 1);
        m_ds = splits[0];
        boost::shared_ptr<ParameterInput> input( new ParameterInput(cuv::extents[bs][m_ds.train_data.shape(1)],"input"));
        boost::shared_ptr<ParameterInput> target( new ParameterInput(cuv::extents[bs][m_ds.train_labels.shape(1)],"target"));
        logistic_regression lr(input, target);
        std::vector<Op*> params = lr.params();

        monitor mon(false);
        mon.add(monitor::WP_SCALAR_EPOCH_STATS, lr.get_loss(), "total loss");
        mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, lr.classification_error(), "classification error");

        m_data   = m_ds.train_data;
        m_labels = m_ds.train_labels;

        gradient_descent gd(lr.get_loss(),0,params, learnrate, -wd);
        m_mon = &mon; m_gd = &gd; m_loss = lr.get_loss();

        gd.before_batch.connect(boost::bind(load_batch,input, target,&m_data, &m_labels, bs,_2));
        gd.current_batch_num = boost::bind(&hyperopt_client::n_batches, this, bs);

        early_stopper es(gd, boost::bind(&monitor::mean, &mon, "classification error"), 1.f, 5, 2.f);
        es.before_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_VALID, 0));
        es.after_early_stopping_epoch.connect(1,boost::bind(&monitor::set_training_phase,&mon, CM_TRAIN, 0));

        mon.register_gd(gd, es);

        gd.minibatch_learning(2, 60*60); // 10 minutes maximum
        
        m_n_epochs = gd.epoch_of_saved_params();
        std::cout << "gd.best_perf():" << es.best_perf() << std::endl;
        return es.best_perf();
    }

    /**
     * Handle a mongodb task specified by the hyper parameters.
     * This method is called from the framework.
     * @param o the task specification
     */
    void handle_task(const mongo::BSONObj& o){                                               
        float lr = o["vals"]["lr"].Array()[0].Double();
        float wd = o["vals"]["wd"].Array()[0].Double();
        int   bs = o["vals"]["bs"].Array()[0].Int();

        bs = pow(2, bs);

        finish(BSON(  "status"  << "ok"
                    <<"loss"    << loss(lr,wd,bs)
                    <<"n_epochs"<< m_n_epochs));
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

