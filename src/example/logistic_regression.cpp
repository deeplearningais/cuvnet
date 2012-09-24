// vim:ts=4:sw=4:et
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <tools/visualization.hpp>
#include <tools/gradient_descent.hpp>
#include <datasets/mnist.hpp>

#include <tools/preprocess.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <tools/monitor.hpp>
#include <tools/logging.hpp>

using namespace cuvnet;
namespace ll = boost::lambda;


/// convenient transpose for a matrix (used in visualization only)
matrix trans(matrix& m){
    matrix mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}


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
 * collects weights, inputs and decoded data from the network and plots it, and
 * writes the results to a file.
 *
 */
void visualize_filters(logistic_regression* lr, monitor* mon, int fa,int fb, int image_size, int channels, boost::shared_ptr<ParameterInput> input, unsigned int epoch){
    if(epoch%50 != 0)
        return;
    {
        std::string base = (boost::format("weights-%06d-")%epoch).str();
        cuv::tensor<float,cuv::host_memory_space>  w = trans(lr->get_weights()->data());
        std::cout << "Weight dims: "<<w.shape(0)<<", "<<w.shape(1)<<std::endl;
        auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
    }
    {
        std::string base = (boost::format("input-%06d-")%epoch).str();
        cuv::tensor<float,cuv::host_memory_space> w = input->data().copy();
        fa = sqrt(w.shape(0));
        fb = sqrt(w.shape(0));
        auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
    }
}
int main(int argc, char **argv)
{
    // initialize cuv library   cuv::initCUDA(0);
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;

    // load the dataset
    mnist_dataset ds("/home/local/datasets/MNIST");
   
    // number of filters is fa*fb (fa and fb determine layout of plots printed
    //          in \c visualize_filters)
    // batch size is bs
    unsigned int fa=2,fb=5,bs=64;

    // makes the values between 0 and 1. Max element of the data will have value 1, and min element valaue 0.
    global_min_max_normalize<> n;
    n.fit_transform(ds.train_data);
    n.transform(ds.test_data);

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input and target correctly
    // \c load_batch will put values in it.
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(1)],"input"));
    boost::shared_ptr<ParameterInput> target(
            new ParameterInput(cuv::extents[bs][ds.train_labels.shape(1)],"target"));

    // creates the logistic regression 
    logistic_regression lr(input, target);

    std::vector<Op*> params = lr.params();

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
        mon.register_gd(gd);

        // after each epoch, run \c visualize_filters
        gd.after_epoch.connect(boost::bind(visualize_filters,&lr,&mon,fa,fb,ds.image_size,ds.channels, input,_1));
        
        // before each batch, load data into \c input
        gd.before_batch.connect(boost::bind(load_batch,input, target,&train_data, &train_labels, bs,_2));
        
        // the number of batches is constant in our case (but has to be supplied as a function)
        gd.current_batch_num = ds.train_data.shape(0)/ll::constant(bs);
        
        // do mini-batch learning for at most 100 epochs, or 10 minutes
        // (whatever comes first)
        gd.minibatch_learning(100, 10*60); // 10 minutes maximum
    }
    std::cout << std::endl << " Test phase: " << std::endl;

    // evaluates test data. We use minibatch learning with learning rate zero and only one epoch.
    {
        matrix train_data = ds.test_data;
        matrix train_labels(ds.test_labels);
        gradient_descent gd(lr.get_loss(),0,params,0.f);
        mon.register_gd(gd);

        gd.before_batch.connect(boost::bind(load_batch,input, target,&train_data, &train_labels, bs,_2));
        gd.current_batch_num = ds.test_data.shape(0)/ll::constant(bs);
        gd.minibatch_learning(1);
    }

    std::cout <<  std::endl;

     
    return 0;
}





