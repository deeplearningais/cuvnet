#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/monitor.hpp>

#include <datasets/random_translation.hpp>
#include "cuvnet/models/relational_auto_encoder.hpp"

using namespace cuvnet;
namespace ll = boost::lambda;

/**
 * load a batch from the dataset
 */
void load_batch(
        boost::shared_ptr<ParameterInput> input_x,
        boost::shared_ptr<ParameterInput> input_y,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        unsigned int bs, unsigned int batch){
    std::cout <<"."<<std::endl;
    input_x->data() = (*data)[cuv::indices[0][cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    input_y->data() = (*data)[cuv::indices[1][cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
}


int main(int argc, char **argv)
{
    // initialize cuv library
    cuv::initCUDA(2);
    cuv::initialize_mersenne_twister_seeds();

    // generate random translation dataset
    std::cout << "generating dataset: "<<std::endl;
    random_translation ds(100, 512*5, 1024, 0.5f, 3, 2.f, 5, 50);
    ds.binary   = false;

    // number of filters is fa*fb (fa and fb determine layout of plots printed
    //          in \c visualize_filters)
    // batch size is bs
    unsigned int fa=16,fb=8,bs=512;

    std::cout << "Traindata: "<<std::endl;
    std::cout << ds.train_data.shape(0)<<std::endl;
    std::cout << ds.train_data.shape(1)<<std::endl;
    std::cout << ds.train_data.shape(2)<<std::endl;
    

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input correctly
    // \c load_batch will put values in it.
    boost::shared_ptr<ParameterInput> input_x(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(2)],"input_x")); 

    boost::shared_ptr<ParameterInput> input_y(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(2)],"input_y")); 

    relational_auto_encoder ae(10, 10, ds.binary); // creates simple autoencoder
    ae.init(input_x, input_y);
    

    // obtain the parameters which we need to derive for in the unsupervised
    // learning phase
    std::vector<Op*> params = ae.unsupervised_params();

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, ae.loss(),        "total loss");
    mon.add(monitor::WP_SINK,               ae.get_decoded_x(), "decoded x");

    // copy training data to the device
    matrix train_data = ds.train_data;

    // create a \c gradient_descent object that derives the auto-encoder loss
    // w.r.t. \c params and has learning rate 0.001f
    gradient_descent gd(ae.loss(),0,params,0.001f);

    // register the monitor so that it receives learning events
    gd.register_monitor(mon);

    gd.after_batch.connect(boost::bind(&monitor::simple_logging, &mon));
    gd.after_batch.connect(std::cout << ll::constant("\n"));

    // before each batch, load data into \c input
    gd.before_batch.connect(boost::bind(load_batch,input_x, input_y,&train_data,bs,_2));

    // the number of batches is constant in our case (but has to be supplied as a function)
    gd.current_batch_num.connect(ds.train_data.shape(1)/ll::constant(bs));

    // do mini-batch learning for at most 6000 epochs, or 10 minutes
    // (whatever comes first)
    gd.minibatch_learning(5000, 10*60); // 10 minutes maximum
    
    return 0;
}
