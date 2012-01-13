// vim:ts=4:sw=4:et
#include <fstream>
#include <cmath>
#include <boost/assign.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <cuvnet/ops.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <cuv/tensor_ops/rprop.hpp>
#include <tools/visualization.hpp>
#include <datasets/cifar.hpp>
#include <datasets/mnist.hpp>

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

    auto_encoder(unsigned int inp0, unsigned int inp1, unsigned int hl, float std=0.1f)
    :m_input(new Input(cuv::extents[inp0][inp1],"input"))
    ,m_weights(new Input(cuv::extents[inp1][hl],"weights"))
    ,m_bias_h(new Input(cuv::extents[hl],       "bias_h"))
    ,m_bias_y(new Input(cuv::extents[inp1],     "bias_y"))
    {
        Op::op_ptr corrupt = std==0.f ? m_input : zero_out(m_input,std);
        m_enc = logistic(mat_plus_vec(
                   prod(
                       corrupt,
                       m_weights)
                   ,m_bias_h,1));
        m_decode = /*logistic*/( mat_plus_vec(
                   prod(
                       m_enc,
                       m_weights, 'n','t')
                   ,m_bias_y,1));

        m_rec_loss    = mean( pow( axpby(m_input, -1.f, m_decode), 2.f)); // reconstruction loss (squared diff)
        //m_rec_loss    = mean( -sum(m_input*log(m_decode) + (1.f-m_input)*log(1.f-m_decode), 1)); // reconstruction loss (cross-entropy)
        m_out         = make_shared<Output>(m_rec_loss->result()); // reconstruction error
        m_reconstruct = make_shared<Output>(m_decode->result());   // for visualization of reconstructed images
        //m_corrupt = make_shared<Output>(corrupt->result());

        m_contractive_loss = 
           sum(sum(pow(m_enc*(1.f-m_enc),2.f),0) * sum(pow(m_weights,2.f),0));
        m_loss        = axpby(m_rec_loss, 0.01f/(float)inp0, m_contractive_loss);
        //m_loss        = m_rec_loss;

        // initialize weights and biases
        float diff = 4.f*std::sqrt(6.f/(inp1+hl));
        cuv::fill_rnd_uniform(m_weights->data());
        m_weights->data() *= 2*diff;
        m_weights->data() -=   diff;
        m_bias_h->data()   = 0.f;
        m_bias_y->data()   = 0.f;
    }
};

int main(int argc, char **argv)
{
    // TODO: TODO TODO: Saxpy functor for cuv!
    cuv::initialize_mersenne_twister_seeds();

#if 0
    {   // check auto-encoder derivatives
        auto_encoder ae(20,40,20,0.0f);
        derivative_tester_verbose(*ae.m_enc);
        derivative_tester_verbose(*ae.m_decode);
        derivative_tester_verbose(*ae.m_rec_loss);
        //derivative_tester_verbose(*ae.m_contractive_loss);
    }
#endif

    mnist_dataset ds("/home/local/datasets/MNIST");
    //cifar_dataset ds;
    unsigned int fa=15,fb=15,bs=20;


#define ONLINE 0
#if ONLINE
    auto_encoder ae(bs,ds.val_data.shape(1),fa*fb,0.00f);
#else
    auto_encoder ae(ds.val_data.shape(0),ds.val_data.shape(1),fa*fb,0.00f);
#endif
    //cuv::libs::cimg::show(arrange_filters(ae.m_weights->data(), 't', fa, fb, ds.image_size,ds.channels), "weights");


    // zmuv normalization
    //ds.val_data -= cuv::mean(ds.val_data);
    //ds.val_data /= std::sqrt(cuv::var(ds.val_data));

    // interval normalization -1, 1
    //ds.val_data -= cuv::minimum(ds.val_data);
    //ds.val_data *= 2.f/cuv::maximum(ds.val_data);
    //ds.val_data -= 1.f;

    // interval normalization 0, 1
    ds.val_data -= cuv::minimum(ds.val_data);
    ds.val_data *= 1.f/cuv::maximum(ds.val_data);
    
    std::vector<Op*> params;
    params += ae.m_weights.get(), ae.m_bias_y.get(), ae.m_bias_h.get();
    swiper swipe(*ae.m_loss,true,params);

    Op::value_type d_old  = ae.m_weights->data(); d_old = 0.f;
    Op::value_type d_oldh = ae.m_bias_h->data(); d_oldh = 0.f;
    Op::value_type d_oldy = ae.m_bias_y->data(); d_oldy = 0.f;
    Op::value_type lr    = ae.m_weights->data(); lr    = 0.0001f;
    Op::value_type lrh   = ae.m_bias_h->data(); lrh   = 0.0001f;
    Op::value_type lry   = ae.m_bias_y->data(); lry   = 0.0001f;

#if ONLINE
    for(unsigned int epoch=0;epoch<25;epoch++){
        for(unsigned int batch=0;batch<ds.val_data.shape(0)/bs; batch++){
            //ae.input().set_view(cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()], ds.val_data);
            ae.input() = cuv::tensor<float,cuv::host_memory_space>(cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()], ds.val_data);
            assert(ae.input().shape(0)==bs);
            swipe.fprop();
            //cuv::libs::cimg::show(arrange_filters(ae.m_corrupt->cdata(),'n', (int)sqrt(bs),(int)sqrt(bs), ds.image_size,ds.channels), "input");
            //std::ofstream os("swiper-fprop.dot");
            //write_graphviz(*ae.m_decode,os,swipe.m_topo.plist);

            swipe.bprop();

            Op::value_type dW = -ae.m_weights->result()->delta.cdata();
            Op::value_type dh = -ae.m_bias_h->result()->delta.cdata();
            Op::value_type dy = -ae.m_bias_y->result()->delta.cdata();
            //cuv::rprop(ae.m_weights->data(), dW, d_old, lr, 0, 0.0000000f);
            //cuv::rprop(ae.m_bias_h->data(), dh, d_oldh, lrh, 0, 0.000000f);
            //cuv::rprop(ae.m_bias_y->data(), dy, d_oldy, lry, 0, 0.000000f);

            //dW += 0.90f*d_old;
            //dh += 0.90f*d_oldh;
            //dy += 0.90f*d_oldy;
            cuv::learn_step_weight_decay( ae.m_weights->data(), dW, .1f);
            cuv::learn_step_weight_decay( ae.m_bias_h->data(),  dh, .1f);
            cuv::learn_step_weight_decay( ae.m_bias_y->data(),  dy, .1f);
            //d_old  = dW;
            //d_oldh = dh;
            //d_oldy = dy;
        }
        std::cout << epoch << " "<<std::sqrt(ae.output()[0])<<" "<<cuv::norm2(ae.m_weights->data())<<" "
            <<cuv::norm2(ae.m_weights->result()->delta.cdata())<<" "
            <<std::sqrt(cuv::var(ae.m_reconstruct->cdata()))<<" "
            <<cuv::norm2(d_old)<<" "
            <<cuv::count(ae.m_weights->data(),0.f)/(float)ae.m_weights->data().size()
            <<std::endl;
    }
#else
    for(unsigned int epoch=0;epoch<1500;epoch++){
            ae.input() = ds.val_data;
            swipe.fprop();
            ae.m_loss->result()->delta.reset(new Op::value_type(ae.m_loss->result()->shape));
            *ae.m_loss->result()->delta = 1.f;
            swipe.bprop();

            std::cout << epoch << " "<<std::sqrt(ae.output()[0])<<" "<<cuv::norm2(ae.m_weights->data())<<" "
                <<cuv::norm2(ae.m_weights->result()->delta.cdata())<<" "
                <<std::sqrt(cuv::var(ae.m_reconstruct->cdata()))<<" "
                <<cuv::norm2(d_old)<<" "
                <<cuv::count(ae.m_weights->data(),0.f)/(float)ae.m_weights->data().size()
                <<std::endl;

            Op::value_type dW = -ae.m_weights->result()->delta.cdata();
            Op::value_type dh = -ae.m_bias_h->result()->delta.cdata();
            Op::value_type dy = -ae.m_bias_y->result()->delta.cdata();
            cuv::rprop(ae.m_weights->data(), dW, d_old, lr, 0, 0.0000000f);
            cuv::rprop(ae.m_bias_h->data(), dh, d_oldh, lrh, 0, 0.000000f);
            cuv::rprop(ae.m_bias_y->data(), dy, d_oldy, lry, 0, 0.000000f);
    }
#endif
    cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', (int)sqrt(bs),(int)sqrt(bs), ds.image_size,ds.channels), "input");
    auto wvis = arrange_filters(ae.m_weights->data(), 't', fa, fb, ds.image_size,ds.channels,true);
    cuv::libs::cimg::save(wvis, "contractive-weights.png");
    return 0;
}
