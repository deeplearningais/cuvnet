// vim:ts=4:sw=4:et
#include <signal.h>
#include <fstream>
#include <cmath>
#include <boost/assign.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <cuvnet/ops.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <cuv/tensor_ops/rprop.hpp>
#include <tools/visualization.hpp>
#include <tools/preprocess.hpp>
#include <tools/gradient_descent.hpp>
#include <datasets/mnist.hpp>
#include <datasets/natural.hpp>
#include <datasets/splitter.hpp>
#include <tools/monitor.hpp>

/// for simple_auto_encoder_no_regularization
#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/convolutional_auto_encoder.hpp>

using namespace boost::assign;
namespace ll = boost::lambda;
typedef denoising_conv_auto_encoder<simple_auto_encoder_no_regularization> ae_type;

/// convenient transpose for a matrix (used in visualization only)
template<class T>
T trans(T& m){
    T mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}

/**
 * load a batch from the dataset
 * In contrast to the `regular' auto_enc example, we have to ensure that inputs
 * have a 3-dimensional shape: batchSize x inputMaps x imagePixels
 */
void load_batch(
        boost::shared_ptr<Input> input,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    input->data() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    input->data().reshape(cuv::extents[input->data().shape(0)][1][input->data().shape(1)]);
}


/**
 * collects weights, inputs and decoded data from the network and plots it, and
 * writes the results to a file.
 *
 */
void visualize_filters(ae_type* ae, monitor* mon, pca_whitening* normalizer, int fa,int fb, int image_size, int channels, boost::shared_ptr<Input> input, unsigned int epoch){
   if(epoch%20 != 0)
       return;
   {
      std::string base = (boost::format("weights-%06d-")%epoch).str();
      cuv::tensor<float,cuv::host_memory_space>  w = ae->get_weights()->data();
      std::cout << "Weight dims: "<<w.shape(0)<<", "<<w.shape(1)<<", "<<w.shape(2)<<std::endl;
      cuvAssert(w.ndim()==3);
      cuvAssert(w.shape(0)==1);
      w.reshape(cuv::extents[w.shape(1)][w.shape(2)]);
      w = trans(w);
      
      auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
      cuv::libs::cimg::save(wvis, base+"nb.png");
      wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
      cuv::libs::cimg::save(wvis, base+"sb.png");
      //normalizer->reverse_transform(w,true); // no mean added
      //wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
      //cuv::libs::cimg::save(wvis, base+"nr.png");
      //wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
      //cuv::libs::cimg::save(wvis, base+"sr.png");
   }

   {
       std::string base = (boost::format("recons-%06d-")%epoch).str();
       cuv::tensor<float,cuv::host_memory_space> w = (*mon)["decoded"].copy();
       w.reshape(cuv::extents[w.shape(0)][w.shape(2)]);
       fa = sqrt(w.shape(0));
       fb = sqrt(w.shape(0));
       auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
       cuv::libs::cimg::save(wvis, base+"nb.png");
       wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
       cuv::libs::cimg::save(wvis, base+"sb.png");
       normalizer->reverse_transform(w); 
       wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
       cuv::libs::cimg::save(wvis, base+"nr.png");
       wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
       cuv::libs::cimg::save(wvis, base+"sr.png");
   }

   {
       std::string base = (boost::format("input-%06d-")%epoch).str();
       cuv::tensor<float,cuv::host_memory_space> w = input->data().copy();
       cuvAssert(w.shape(1)==1);
       w.reshape(cuv::extents[w.shape(0)][w.shape(2)]);
       fa = sqrt(w.shape(0));
       fb = sqrt(w.shape(0));
       auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
       cuv::libs::cimg::save(wvis, base+"nb.png");
       wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
       cuv::libs::cimg::save(wvis, base+"sb.png");
       normalizer->reverse_transform(w); 
       wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
       cuv::libs::cimg::save(wvis, base+"nr.png");
       wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
       cuv::libs::cimg::save(wvis, base+"sr.png");
   }
}

int main(int argc, char **argv)
{
    // initialize cuv library
    cuv::initCUDA(1);
    cuv::initialize_mersenne_twister_seeds();

    //mnist_dataset ds_all("/home/local/datasets/MNIST");
    natural_dataset ds_all("/home/local/datasets/natural_images");
    pca_whitening normalizer(128,false,true, 0.01);
    splitter ds_split(ds_all,2);
    dataset ds  = ds_split[0];
    ds.binary   = false;

    // number of filters is fa*fb (fa and fb determine layout of plots printed in \c visualize_filters)
    // batch size is bs
    unsigned int fa=8,fb=8,bs=64;
    
    {   //-------------------------------------------------------------
        // pre-processing                                              +
        
        // subtract mean of each patch
        // http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing

        // Osindero & Hinton: First log, then ZMUV, then ZCA.
        // remark: weird: higher order whitening paper (CVPR'05) also whitens
        // in log-space /after/ regular whitening, retaining the sign
        preprocessing_pipeline pp;
        pp  .add(new log_transformer<>())
            .add(new zero_sample_mean<>())
            .add(new zero_mean_unit_variance());
        pp.fit_transform(ds.train_data); // normalize each feature to get to defined range
        pp.transform(ds.val_data);    // do the same to the validation set

        normalizer.fit_transform(ds.train_data);
        normalizer.transform(ds.val_data);
        // end preprocessing                                           /
        //-------------------------------------------------------------
    }

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input correctly
    // \c load_batch will put values in it.
    boost::shared_ptr<Input> input(
            new Input(cuv::extents[bs][1][ds.train_data.shape(1)],"input")); 

    // create a denoising convolutional auto-encoder
    // given filter-size, number of filters and noise standard deviation
    ae_type ae(ds.binary, 5, fa*fb, 1.0); 
    ae.init(input, 0.001f);
    
    // obtain the parameters which we need to derive for in the unsupervised
    // learning phase
    std::vector<Op*> params = ae.unsupervised_params();

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true); 
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, ae.loss(),        "total loss");
    mon.add(monitor::WP_SINK, ae.get_decoded(), "decoded");

    // copy training data to the device
    matrix data = ds.train_data;

    // create a \c gradient_descent object that derives the auto-encoder loss
    // w.r.t. \c params and has learning rate 0.0001f
    gradient_descent gd(ae.loss(),0,params,0.0001f);

    // register the monitor so that it receives learning events
    gd.register_monitor(mon); 

    // after each epoch, run \c visualize_filters
    gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&mon,&normalizer,fa,fb,ds.image_size,ds.channels, input,_1));

    // before each batch, load data into \c input
    gd.before_batch.connect(boost::bind(load_batch,input,&data,bs,_2));

    // the number of batches is constant in our case (but has to be supplied as a function)
    gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));

    // do mini-batch learning for at most 6000 epochs, or 10 minutes
    // (whatever comes first)
    gd.minibatch_learning(6000, 10*60); 
    
    return 0;
}