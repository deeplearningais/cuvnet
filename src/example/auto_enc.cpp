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
    boost::shared_ptr<Output> m_out, m_reconstruct;
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
        m_enc = logistic(mat_plus_vec(
                    prod(
                        std==0.f ? m_input : add_rnd_normal(m_input,std),
                        m_weights)
                    ,m_bias_h,1));
        m_decode = logistic( mat_plus_vec(
                    prod(
                        m_enc,
                        m_weights, 'n','t')
                    ,m_bias_y,1)
                ,true);

        m_rec_loss    = mean( pow( axpby(m_input, -1.f, m_decode), 2.f)); // reconstruction loss (squared diff)
        m_out         = make_shared<Output>(m_rec_loss->result()); // reconstruction error
        m_reconstruct = make_shared<Output>(m_decode->result());   // for visualization of reconstructed images

        //m_contractive_loss = 
        //    sum(sum(pow(m_enc*(1.f-m_enc),2.f),0) * sum(pow(m_weights,2.f),0));
        //m_loss        = axpby(m_rec_loss, 0.001f/(float)inp0, m_contractive_loss);
        m_loss        = m_rec_loss;

        // initialize weights and biases
        float diff = 4.f*std::sqrt(6.f/(inp1+hl));
        cuv::fill_rnd_uniform(m_weights->data());
        m_weights->data() *= 2*diff;
        m_weights->data() -=   diff;
        m_bias_h->data() = 0.f;
        m_bias_y->data() = 0.f;
    }
};

int main(int argc, char **argv)
{
    // TODO: TODO TODO: Saxpy functor for cuv!
    cuv::initialize_mersenne_twister_seeds();

    {   // check auto-encoder derivatives
        auto_encoder ae(10,20,5,0.0f);
        derivative_tester_verbose(*ae.m_enc);
        derivative_tester_verbose(*ae.m_decode);
        derivative_tester_verbose(*ae.m_rec_loss);
        derivative_tester_verbose(*ae.m_contractive_loss);
    }
    exit(0);

    mnist_dataset ds("/home/local/datasets/MNIST");
    //cifar_dataset ds;
    unsigned int fa=5,fb=5;


    auto_encoder ae(ds.val_data.shape(0),ds.val_data.shape(1),fa*fb,0.0f);
    //cuv::libs::cimg::show(arrange_filters(ae.m_weights->data(), 't', fa, fb, ds.image_size,ds.channels), "weights");

    ae.input()  = ds.val_data;

    // zmuv normalization
    //ae.input() -= cuv::mean(ae.input());
    //ae.input() /= std::sqrt(cuv::var(ae.input()));

    // interval normalization -1, 1
    //ae.input() -= cuv::minimum(ae.input());
    //ae.input() *= 2.f/cuv::maximum(ae.input());
    //ae.input() -= 1.f;

    // interval normalization 0, 1
    ae.input() -= cuv::minimum(ae.input());
    ae.input() *= 1.f/cuv::maximum(ae.input());
    
    std::vector<Op*> params;
    params += ae.m_weights.get(), ae.m_bias_y.get(), ae.m_bias_h.get();
    swiper swipe(*ae.m_loss,true,params);

    Op::value_type d_old  = ae.m_weights->data(); d_old = 0.f;
    Op::value_type d_oldh = ae.m_bias_h->data(); d_oldh = 0.f;
    Op::value_type d_oldy = ae.m_bias_y->data(); d_oldy = 0.f;
    Op::value_type lr    = ae.m_weights->data(); lr    = 0.0001f;
    Op::value_type lrh   = ae.m_bias_h->data(); lrh   = 0.0001f;
    Op::value_type lry   = ae.m_bias_y->data(); lry   = 0.0001f;

    for(unsigned int epoch=0;epoch<50000;epoch++){
        swipe.fprop();
        //std::ofstream os("swiper-fprop.dot");
        //write_graphviz(*ae.m_decode,os,swipe.m_topo.plist);

        ae.m_loss->result()->delta.reset(new Op::value_type(ae.m_loss->result()->shape));
        *ae.m_loss->result()->delta = 1.f;

        swipe.bprop();
        std::cout << epoch << " "<<std::sqrt(ae.output()[0])<<" "<<cuv::norm2(ae.m_weights->data())<<" "
                                                                 <<cuv::norm2(ae.m_weights->result()->delta.cdata())<<" "
                                                                 <<cuv::count(ae.m_weights->data(),0.f)/(float)ae.m_weights->data().size()
                                                                 <<std::endl;

        //Op::value_type dW = -ae.m_weights->result()->delta.cdata();
        //Op::value_type dh = -ae.m_bias_h->result()->delta.cdata();
        //Op::value_type dy = -ae.m_bias_y->result()->delta.cdata();
        //cuv::rprop(ae.m_weights->data(), dW, d_old, lr, 0, 0.0000001f);
        //cuv::rprop(ae.m_bias_h->data(), dh, d_oldh, lrh, 0, 0.000000f);
        //cuv::rprop(ae.m_bias_y->data(), dy, d_oldy, lry, 0, 0.000000f);
        cuv::learn_step_weight_decay( ae.m_weights->data(), const_cast<matrix&>(ae.m_weights->result()->delta.cdata()), -.01f,.01f);
        cuv::learn_step_weight_decay( ae.m_bias_h->data(),  const_cast<matrix&>(ae.m_bias_h->result()->delta.cdata()), -.01f);
        cuv::learn_step_weight_decay( ae.m_bias_y->data(),  const_cast<matrix&>(ae.m_bias_y->result()->delta.cdata()), -.01f);
    }
    //cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', fa,fb, ds.image_size,ds.channels), "input");
    auto wvis = arrange_filters(ae.m_weights->data(), 't', fa, fb, ds.image_size,ds.channels);
    cuv::libs::cimg::save(wvis, "contractive-weights.png");
    return 0;
}
