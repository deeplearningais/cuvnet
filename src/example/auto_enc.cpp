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
#include <datasets/cifar.hpp>
#include <datasets/mnist.hpp>
#include <datasets/natural.hpp>
#include <datasets/amat_datasets.hpp>
#include <datasets/splitter.hpp>
#include <tools/monitor.hpp>

#include "cuvnet/models/simple_auto_encoder.hpp"

using namespace boost::assign;
namespace ll = boost::lambda;
typedef simple_auto_encoder<simple_auto_encoder_weight_decay> ae_type;

matrix trans(matrix& m){
    matrix mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}

void load_batch(
        boost::shared_ptr<Input> input,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    input->data() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
}


void visualize_filters(ae_type* ae, monitor* mon, pca_whitening* normalizer, int fa,int fb, int image_size, int channels, boost::shared_ptr<Input> input, unsigned int epoch){
    if(epoch%300 != 0)
        return;
    {
        std::string base = (boost::format("weights-%06d-")%epoch).str();
        // show the resulting filters
        //unsigned int n_rec = (bs>0) ? sqrt(bs) : 6;
        //cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', n_rec,n_rec, image_size,channels), "input");
        cuv::tensor<float,cuv::host_memory_space>  w = trans(ae->get_weights()->data());
        std::cout << "Weight dims: "<<w.shape(0)<<", "<<w.shape(1)<<std::endl;
        auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
        normalizer->reverse_transform(w,true); // no mean added
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
        cuv::libs::cimg::save(wvis, base+"nr.png");
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
        cuv::libs::cimg::save(wvis, base+"sr.png");
    }

    {
        std::string base = (boost::format("recons-%06d-")%epoch).str();
        cuv::tensor<float,cuv::host_memory_space> w = (*mon)["decoded"].copy();
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
    cuv::initCUDA(2);
    cuv::initialize_mersenne_twister_seeds();

    //mnist_dataset ds_all("/home/local/datasets/MNIST");
    natural_dataset ds_all("/home/local/datasets/natural_images");
    pca_whitening normalizer(128,true,true, 0.01);
    //global_min_max_normalize<> normalizer(0,1); // 0,1
    //cifar_dataset ds;
    //zero_mean_unit_variance<> normalizer;
    //amat_dataset ds_all("/home/local/datasets/bengio/mnist.zip","mnist_train.amat", "mnist_test.amat");
    //global_min_max_normalize<> normalizer(0,1); // 0,1
    splitter ds_split(ds_all,2);
    dataset ds  = ds_split[0];
    ds.binary   = false;

    unsigned int fa=16,fb=8,bs=512;
    
    {   //-------------------------------------------------------------
        // pre-processing                                              +
        
        // subtract mean of each patch
        // http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing

        // Osindero & Hinton: First log, then ZMUV, then ZCA.
        // remark: weird: higher order whitening paper (CVPR'05) also whitens
        // in log-space /after/ regular whitening, retaining the sign
        zero_sample_mean<> n;
        log_transformer<> n2;
        zero_mean_unit_variance<> n3; 

        n2.fit_transform(ds.train_data); // results pretty much in gaussian
        n2.transform(ds.val_data);    // do the same to the validation set

        n.fit_transform(ds.train_data); // subtract sample mean
        n.transform(ds.val_data);

        n3.fit_transform(ds.train_data); // normalize each feature to get to defined range
        n3.transform(ds.val_data);    // do the same to the validation set
        

        normalizer.fit_transform(ds.train_data);
        normalizer.transform(ds.val_data);
        // end preprocessing                                           /
        //-------------------------------------------------------------
    }

    boost::shared_ptr<Input> input(
            new Input(cuv::extents[bs][ds.train_data.shape(1)],"input")); 
    ae_type ae(fa*fb, ds.binary); // creates simple autoencoder
    ae.init(input, 0.001f);
    

    std::vector<Op*> params = ae.unsupervised_params();

    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, ae.loss(),        "total loss");
    mon.add(monitor::WP_SINK,               ae.get_decoded(), "decoded");

    gradient_descent gd(ae.loss(),0,params,0.001f);
    gd.register_monitor(mon);
    gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&mon,&normalizer,fa,fb,ds.image_size,ds.channels, input,_1));
    gd.before_batch.connect(boost::bind(load_batch,input,&ds.train_data,bs,_2));
    gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));
    gd.minibatch_learning(6000, 10*60); // 10 minutes maximum
    
    return 0;
}
