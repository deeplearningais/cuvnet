// vim:ts=4:sw=4:et
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <tools/visualization.hpp>
#include <tools/gradient_descent.hpp>
#include <datasets/mnist.hpp>

#include <tools/preprocess.hpp>
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <tools/monitor.hpp>

using namespace boost::assign;
using namespace cuvnet;
namespace ll = boost::lambda;

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

int main(int argc, char **argv)
{
    // initialize cuv library   cuv::initCUDA(0);
    cuv::initCUDA(0);
    cuv::initialize_mersenne_twister_seeds();

    // load the dataset
    mnist_dataset ds("/home/local/datasets/MNIST");
    
    // number of filters is fa*fb (fa and fb determine layout of plots printed
    //          in \c visualize_filters)
    // batch size is bs
    unsigned int fa=16,fb=8,bs=64;

    // makes the values between 0 and 1. Max element of the data will have value 1, and min element valaue 0.
    global_min_max_normalize<> n;
    n.fit_transform(ds.train_data);

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input and target correctly
    // \c load_batch will put values in it.
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(1)],"input"));
    boost::shared_ptr<ParameterInput> target(
            new ParameterInput(cuv::extents[bs][ds.train_labels.shape(1)],"target"));

    //creates stacked autoencoder with one simple autoencoder. has fa*fb number of hidden units
    auto_encoder_stack ae_s(ds.binary);
    typedef l2reg_simple_auto_encoder ae_type;
    ae_s.add<ae_type>(fa*fb, ds.binary, 0.01f);
    ae_s.init(input);

    // creates the logistic regression on the top of the stacked autoencoder
    //linear_regression lr(ae_s.get_encoded(), target);
    logistic_regression lr(ae_s.get_encoded(), target);

    // puts the supervised parameters of the stacked autoencoder and logistic regression parameters in one vector
    std::vector<Op*> params = ae_s.supervised_params();
    std::vector<Op*> params_log = lr.params();
    std::copy(params_log.begin(), params_log.end(), std::back_inserter(params));

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, lr.get_loss(),        "total loss");
    mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, lr.classification_error(),        "classification error");

    // copy training data and labels to the device, and converts train_labels from int to float
    matrix train_data = ds.train_data;
    matrix train_labels(ds.train_labels);
    
    std::cout << std::endl << " Training phase: " << std::endl;
    {
        // create a \c gradient_descent object that derives the logistic loss
        // w.r.t. \c params and has learning rate 0.1f
        gradient_descent gd(lr.get_loss(),0,params,0.1f);
        
        // register the monitor so that it receives learning events
        gd.register_monitor(mon);
        
        // after each epoch, run \c visualize_filters
        //gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&normalizer,fa,fb,ds.image_size,ds.channels, input,_1));
        
        // before each batch, load data into \c input
        gd.before_batch.connect(boost::bind(load_batch,input, target,&train_data, &train_labels, bs,_2));
        
        // the number of batches is constant in our case (but has to be supplied as a function)
        gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));
        
        // do mini-batch learning for at most 100 epochs, or 10 minutes
        // (whatever comes first)
        gd.minibatch_learning(100, 10*60); // 10 minutes maximum
    }
    std::cout << std::endl << " Test phase: " << std::endl;

    // evaluates test data, does it similary as with train data, minibatch_learing is running only one epoch
    {
        matrix train_data = ds.test_data;
        matrix train_labels(ds.test_labels);
        gradient_descent gd(lr.get_loss(),0,params,0.f);
        gd.register_monitor(mon);
        gd.before_batch.connect(boost::bind(load_batch,input, target,&train_data, &train_labels, bs,_2));
        gd.current_batch_num.connect(ds.test_data.shape(0)/ll::constant(bs));
        gd.minibatch_learning(1);
    }

    std::cout <<  std::endl;

     
    return 0;
}

