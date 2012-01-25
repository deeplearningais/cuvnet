// vim:ts=4:sw=4:et
#include <signal.h>
#include <fstream>
#include <cmath>
#include <boost/assign.hpp>
#include <boost/bind.hpp>
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
#include <datasets/amat_datasets.hpp>
#include <datasets/splitter.hpp>

using namespace cuvnet;
using namespace boost::assign;
using boost::make_shared;

struct auto_encoder{
    boost::shared_ptr<Input>  m_input, m_weights,m_bias_h,m_bias_y;
    boost::shared_ptr<Output> m_out, m_reconstruct, m_corrupt;
    boost::shared_ptr<Op>     m_decode, m_enc;
    boost::shared_ptr<Op>     m_loss, m_rec_loss, m_contractive_loss;
    matrix&       input() {return m_input->data();}
    const matrix& output(){return m_out->cdata();}

    void print_loss(unsigned int epoch){ std::cout << epoch<<" " << output()[0]<<std::endl;}

    auto_encoder(unsigned int inp0, unsigned int inp1, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
    :m_input(new Input(cuv::extents[inp0][inp1],"input"))
    ,m_weights(new Input(cuv::extents[inp1][hl],"weights"))
    ,m_bias_h(new Input(cuv::extents[hl],       "bias_h"))
    ,m_bias_y(new Input(cuv::extents[inp1],     "bias_y"))
    {
        Op::op_ptr corrupt               = m_input;
        if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
        if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);
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
        m_out         = make_shared<Output>(m_rec_loss->result()); // reconstruction error
        m_reconstruct = make_shared<Output>(m_decode->result());   // for visualization of reconstructed images
        //m_corrupt = make_shared<Output>(corrupt->result());      // for visualization of corrupted     images

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
    ae->input() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
}

int main(int argc, char **argv)
{
    cuv::initialize_mersenne_twister_seeds();
    {   // check auto-encoder derivatives
        auto_encoder ae(2,4,2,false,0.0f,0.2f);
        derivative_tester_verbose(*ae.m_enc);
        derivative_tester_verbose(*ae.m_decode);
        derivative_tester_verbose(*ae.m_rec_loss);
        if(ae.m_contractive_loss.get()){
            derivative_tester_verbose(*ae.m_contractive_loss);
        }
    }

    mnist_dataset ds_all("/home/local/datasets/MNIST");
    global_min_max_normalize<> normalizer(0,1); // 0,1
    //cifar_dataset ds;
    //zero_mean_unit_variance<> normalizer;
    //amat_dataset ds_all("/home/local/datasets/bengio/mnist.zip","mnist_train.amat", "mnist_test.amat");
    //global_min_max_normalize<> normalizer(0,1); // 0,1
    splitter ds_split(ds_all,10);
    dataset& ds  = ds_split[0];
    
    normalizer.fit_transform(ds.train_data);
    normalizer.transform(ds.val_data);

    unsigned int fa=16,fb=16,bs=64;
    auto_encoder ae(bs==0?ds.val_data.shape(0):bs,
            ds.train_data.shape(1), fa*fb, 
            ds.channels==1, 0.00f, 0.050000f); // CIFAR: lambda=0.05, MNIST lambda=1.0

    std::vector<Op*> params;
    params += ae.m_weights.get(), ae.m_bias_y.get(), ae.m_bias_h.get();

    Op::value_type alldata = bs==0 ? ds.val_data : ds.train_data;
    gradient_descent gd(ae.m_loss,params,0.1f,0.00000f);
    gd.after_epoch.connect(boost::bind(&auto_encoder::print_loss, &ae, _1));
    gd.before_batch.connect(boost::bind(load_batch,&ae,&alldata,bs,_2));
    if(bs==0){
        ae.input()= alldata;
        alldata.dealloc();
        gd.batch_learning(3200);
    }
    else      gd.minibatch_learning(400, ds.train_data.shape(0)/bs);
    
    // show the resulting filters
    //unsigned int n_rec = (bs>0) ? sqrt(bs) : 6;
    //cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', n_rec,n_rec, ds.image_size,ds.channels), "input");
    auto wvis = arrange_filters(ae.m_weights->data(), 't', fa, fb, ds.image_size,ds.channels,false);
    cuv::libs::cimg::save(wvis, "contractive-weights.png");
    wvis      = arrange_filters(ae.m_weights->data(), 't', fa, fb, ds.image_size,ds.channels,true);
    cuv::libs::cimg::save(wvis, "contractive-weights-sepn.png");
    return 0;
}
