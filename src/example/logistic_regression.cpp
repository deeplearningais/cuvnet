// vim:ts=4:sw=4:et
#include <signal.h>
#include <fstream>
#include <cmath>
#include <boost/assign.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/export.hpp>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <tools/visualization.hpp>
#include <tools/gradient_descent.hpp>
#include <mongo/bson/bson.h>
#include <datasets/mnist.hpp>
#include <tools/crossvalid.hpp>
#include <tools/learner.hpp>

#include <tools/preprocess.hpp>
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/generic_regression.hpp>
#include <tools/monitor.hpp>

using namespace boost::assign;
using namespace cuvnet;
namespace ll = boost::lambda;

void load_batch(
        boost::shared_ptr<Input> input,
        boost::shared_ptr<Input> target,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        cuv::tensor<float,cuv::dev_memory_space>* labels,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    input->data() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    target->data() = (*labels)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];

}

int main(int argc, char **argv)
{
    cuv::initCUDA(0);
    cuv::initialize_mersenne_twister_seeds();

    mnist_dataset ds("/home/local/datasets/MNIST");
    unsigned int fa=16,fb=16,bs=64;

    global_min_max_normalize<> n;
    n.fit_transform(ds.train_data);

    // inits input and target
    boost::shared_ptr<Input> input(
            new Input(cuv::extents[bs][ds.train_data.shape(1)],"input"));
    boost::shared_ptr<Input> target(
            new Input(cuv::extents[bs][ds.train_labels.shape(1)],"target"));

    //creates stacked autoencoder with one simple autoencoder
    auto_encoder_stack<> ae_s(ds.binary);
    typedef simple_auto_encoder<simple_auto_encoder_weight_decay> ae_type;
    ae_s.add<ae_type>(fa*fb, ds.binary);
    ae_s.init(input, 0.01f);

    // creates the logistic regression on the top of the stacked autoencoder
    //linear_regression lr(ae_s.get_encoded(), target);
    logistic_regression lr(ae_s.get_encoded(), target);

    std::vector<Op*> params = ae_s.supervised_params();
    std::vector<Op*> params_log = lr.params();
    std::copy(params_log.begin(), params_log.end(), std::back_inserter(params));

    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, lr.get_loss(),        "total loss");
    mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, lr.classification_error(),        "classification error");

    matrix train_data = ds.train_data;
    matrix train_labels(ds.train_labels.shape());
    {
        cuv::tensor<int,cuv::dev_memory_space>  tmp(ds.train_labels);
        cuv::convert(train_labels, tmp);
    }
    
    std::cout << std::endl << " Training phase: " << std::endl;
    {
        gradient_descent gd(lr.get_loss(),0,params,0.1f);
        gd.register_monitor(mon);
        //gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&normalizer,fa,fb,ds.image_size,ds.channels, input,_1));
        gd.before_batch.connect(boost::bind(load_batch,input, target,&train_data, &train_labels, bs,_2));
        gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));
        gd.minibatch_learning(100, 10*60); // 10 minutes maximum
    }
    std::cout << std::endl << " Test phase: " << std::endl;

    {
        matrix train_data = ds.test_data;
        matrix train_labels(ds.test_labels.shape());
        {
            cuv::tensor<int,cuv::dev_memory_space>  tmp(ds.test_labels);
            cuv::convert(train_labels, tmp);
        }
        gradient_descent gd(lr.get_loss(),0,params,0.f);
        gd.register_monitor(mon);
        gd.before_batch.connect(boost::bind(load_batch,input, target,&train_data, &train_labels, bs,_2));
        gd.current_batch_num.connect(ds.test_data.shape(0)/ll::constant(bs));
        gd.minibatch_learning(1);
    }

    std::cout <<  std::endl;

     
    return 0;
}





