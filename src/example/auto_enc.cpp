// vim:ts=4:sw=4:et
#include <signal.h>
#include <fstream>
#include <cmath>
#include <boost/assign.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

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

using namespace cuvnet;
using namespace boost::assign;
using boost::make_shared;
namespace ll = boost::lambda;
namespace acc=boost::accumulators;

typedef
acc::accumulator_set<double,
    acc::stats<acc::tag::mean, acc::tag::variance(acc::lazy) > > acc_t;

matrix trans(matrix& m){
    matrix mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}

struct auto_encoder{
    boost::shared_ptr<Input>  m_input, m_weights,m_bias_h,m_bias_y;
    boost::shared_ptr<Sink> m_out, m_reconstruct, m_corrupt;
    boost::shared_ptr<Op>     m_decode, m_enc;
    boost::shared_ptr<Op>     m_loss, m_rec_loss, m_contractive_loss;
    acc_t                     s_loss;
    matrix&       input() {return m_input->data();}
    const matrix& output(){return m_out->cdata();}

    void print_loss(unsigned int epoch){ std::cout << epoch<<": "<<acc::mean(s_loss)<<std::endl;}
    void acc_loss(){ s_loss((float) output()[0]);}
    void reset_loss(){ s_loss = acc_t();}

    auto_encoder(unsigned int inp0, unsigned int inp1, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
    :m_input(new Input(cuv::extents[inp0][inp1],"input"))
    ,m_weights(new Input(cuv::extents[inp1][hl],"weights"))
    ,m_bias_h(new Input(cuv::extents[hl],       "bias_h"))
    ,m_bias_y(new Input(cuv::extents[inp1],     "bias_y"))
    {
        Op::op_ptr corrupt;
        if(0);
        else if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
        else if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);
        else corrupt = m_input;
        m_enc    = logistic(mat_plus_vec(
                    prod( corrupt, m_weights)
                    ,m_bias_h,1));
        m_decode = mat_plus_vec(
                prod( m_enc, m_weights, 'n','t')
                ,m_bias_y,1);

        if(!binary)  // squared loss
            m_rec_loss = mean( pow( axpby(m_input, -1.f, m_decode), 2.f)); 
        else         // cross-entropy
            m_rec_loss = mean( sum(neg_log_cross_entropy_of_logistic(m_input,m_decode),1));
        m_out         = make_shared<Sink>(m_rec_loss->result()); // reconstruction error
        m_reconstruct = make_shared<Sink>(m_decode->result());   // for visualization of reconstructed images
        //m_corrupt = make_shared<Sink>(corrupt->result());      // for visualization of corrupted     images

        if(lambda>0.f){ // contractive AE
            m_contractive_loss = 
                sum(sum(pow(m_enc*(1.f-m_enc),2.f),0) 
                        * sum(pow(m_weights,2.f),0));
            m_loss        = axpby(m_rec_loss, lambda/(float)inp0, m_contractive_loss);
        }else
            m_loss        = m_rec_loss; // no change

        // initialize weights and biases
        float diff = 4.f*std::sqrt(6.f/(inp1+hl));
        cuv::fill_rnd_uniform(m_weights->data());
        m_weights->data() *= 2*diff;
        m_weights->data() -=   diff;
        m_bias_h->data()   = 0.f;
        m_bias_y->data()   = 0.f;
    }
};

void load_batch(
        auto_encoder* ae,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    ae->input() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
}

void visualize_filters(auto_encoder* ae, pca_whitening* normalizer, int fa,int fb, int image_size, int channels, unsigned int epoch){
    if(epoch%100 != 0)
        return;
    {
        std::string base = (boost::format("weights-%06d-")%epoch).str();
        // show the resulting filters
        //unsigned int n_rec = (bs>0) ? sqrt(bs) : 6;
        //cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', n_rec,n_rec, image_size,channels), "input");
        cuv::tensor<float,cuv::host_memory_space>  w = trans(ae->m_weights->data());
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
        cuv::tensor<float,cuv::host_memory_space> w = ae->m_reconstruct->cdata().copy();
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
        cuv::tensor<float,cuv::host_memory_space> w = ae->m_input->data().copy();
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
    cuv::initCUDA(1);
    cuv::initialize_mersenne_twister_seeds();
    {   // check auto-encoder derivatives
        auto_encoder ae(2,4,2,false,0.0f,0.2f);
        derivative_tester_verbose(*ae.m_enc,0);
        derivative_tester_verbose(*ae.m_decode,0);
        derivative_tester_verbose(*ae.m_rec_loss,0);
        if(ae.m_contractive_loss.get()){
            derivative_tester_verbose(*ae.m_contractive_loss,0);
        }
    }

    //mnist_dataset ds_all("/home/local/datasets/MNIST");
    natural_dataset ds_all("/home/local/datasets/natural_images");
    pca_whitening normalizer(-1,true,true, 0.1);
    //global_min_max_normalize<> normalizer(0,1); // 0,1
    //cifar_dataset ds;
    //zero_mean_unit_variance<> normalizer;
    //amat_dataset ds_all("/home/local/datasets/bengio/mnist.zip","mnist_train.amat", "mnist_test.amat");
    //global_min_max_normalize<> normalizer(0,1); // 0,1
    splitter ds_split(ds_all,2);
    dataset ds  = ds_split[0];
    ds.binary   = false;

    unsigned int fa=16,fb=16,bs=512;
    
    {   //-------------------------------------------------------------
        // pre-processing                                              +
        
        // subtract mean of each patch
        // http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing
        zero_sample_mean<> n;
        n.fit_transform(ds.train_data);
        n.transform(ds.val_data);

        // Osindero & Hinton: First log, then ZMUV, then ZCA.
        // remark: weird: higher order whitening paper (CVPR'05) also whitens
        // in log-space /after/ regular whitening, retaining the sign
        log_transformer<> n2;
        zero_mean_unit_variance<> n3; 
        //n2.fit_transform(ds.train_data);
        n3.fit_transform(ds.train_data);
        //n2.transform(ds.val_data);
        n3.transform(ds.val_data);

        normalizer.fit_transform(ds.train_data);
        normalizer.transform(ds.val_data);
        // end preprocessing                                           /
        //-------------------------------------------------------------
    }
    
    
    std::cout << "train_data dim: "<<ds.train_data.shape(1)<<std::endl;
    std::cout << "train_data max: "<<cuv::maximum(ds.train_data)<<std::endl;
    std::cout << "train_data min: "<<cuv::minimum(ds.train_data)<<std::endl;
    std::cout << "train_data mean: "<<cuv::mean(ds.train_data)<<std::endl;
    std::cout << "train_data var : "<<cuv::var(ds.train_data)<<std::endl;


    auto_encoder ae(bs==0?ds.val_data.shape(0):bs,
            ds.train_data.shape(1), fa*fb, 
            ds.binary, 0.0f, 0.000000f); // CIFAR: lambda=0.05, MNIST lambda=1.0

    std::vector<Op*> params;
    params += ae.m_weights.get(), ae.m_bias_y.get(), ae.m_bias_h.get();

    Op::value_type alldata = bs==0 ? ds.val_data : ds.train_data;
    gradient_descent gd(ae.m_loss,0,params,0.01f,-0.00010f);
    gd.after_epoch.connect(boost::bind(&auto_encoder::print_loss, &ae, _1));
    gd.after_epoch.connect(boost::bind(&auto_encoder::reset_loss, &ae));
    gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&normalizer,fa,fb,ds.image_size,ds.channels,_1));
    gd.before_batch.connect(boost::bind(load_batch,&ae,&alldata,bs,_2));
    gd.after_batch.connect(boost::bind(&auto_encoder::acc_loss, &ae));
    gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));
    if(bs==0){
        ae.input()= alldata;
        alldata.dealloc();
        gd.batch_learning(3200);
    }
    else      gd.minibatch_learning(6000);
    
    return 0;
}
