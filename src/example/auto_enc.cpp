// vim:ts=4:sw=4:et
#include <fstream>
#include <cmath>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <cuv/tensor_ops/rprop.hpp>
#include <tools/visualization.hpp>
#include <datasets/cifar.hpp>
#include <datasets/mnist.hpp>

using namespace cuvnet;

struct auto_encoder{
    boost::shared_ptr<Input>  m_input;
    boost::shared_ptr<Input>  m_weights;
    boost::shared_ptr<Output> m_out;
    boost::shared_ptr<Op>     m_func;
    matrix&       input() {return m_input->data();}
    const matrix& output(){return m_out->cdata();}

    auto_encoder(unsigned int inp0, unsigned int inp1, unsigned int hl, float std=0.1f)
    :m_input(new Input(cuv::extents[inp0][inp1]))
    ,m_weights(new Input(cuv::extents[inp1][hl]))
    {
        // Sum ( input - tanh( X W ) W' )^2
        m_func = sum( pow( axpby(1.f,m_input, -1.f, prod(
                            tanh(prod(
                                    add_rnd_normal(m_input,std),
                                    m_weights)),
                            m_weights, 'n','t')),
                    2.f));
        m_out = boost::make_shared<Output>(m_func->result());

        float diff = 4.f*std::sqrt(6.f/(inp1+hl));
        float mult = 2.f*diff;
        cuv::tensor<float,cuv::host_memory_space> x = m_weights->data();
        for(unsigned int i=0;i<x.size();i++)
            x[i] = (float)(drand48()*mult-diff);
        m_weights->data() = x;
    }
};

int main(int argc, char **argv)
{
    // TODO: TODO TODO: Saxpy functor for cuv!
    cuv::initialize_mersenne_twister_seeds();
    mnist_dataset ds("/home/local/datasets/MNIST");
    unsigned int fa=15,fb=15;
    auto_encoder ae(ds.val_data.shape(0),ds.val_data.shape(1),fa*fb,0.05f);
    {
        cuv::tensor<float,cuv::host_memory_space> w = ae.m_weights->data();
        cuv::tensor<float,cuv::host_memory_space> wt(cuv::extents[w.shape(1)][w.shape(0)]);
        wt = 0.f;
        cuv::transpose(wt,w);
        wt -= cuv::minimum(wt);
        wt /= cuv::maximum(wt);
        cuv::libs::cimg::show(arrange_filters(wt, fa, fb, ds.image_size,ds.channels), "weights");
    }

    // generate data
    ae.input()  = ds.val_data;
    ae.input() -= cuv::mean(ae.input());
    ae.input() /= std::sqrt(cuv::var(ae.input()));
    
    std::vector<Op*> params(1,ae.m_weights.get());
    swiper swipe(*ae.m_func,true,params);

    for(unsigned int epoch=0;epoch<2000;epoch++){
        swipe.fprop();
        std::cout << epoch << " "<<std::sqrt(ae.output()[0]/ae.input().shape(1)/ae.input().shape(0))<<" "<<cuv::norm2(ae.m_weights->data())<<std::endl;
        swipe.bprop();

        cuv::learn_step_weight_decay(
                ae.m_weights->data(),
                const_cast<matrix&>(ae.m_weights->result()->delta.cdata()),
                -0.001f/ds.val_data.shape(0),.001f);
    }

    cuv::tensor<float,cuv::host_memory_space> w = ae.m_weights->data();
    cuv::tensor<float,cuv::host_memory_space> wt(cuv::extents[w.shape(1)][w.shape(0)]);
    cuv::transpose(wt,w);
    wt -= cuv::minimum(wt);
    wt /= cuv::maximum(wt);
    cuv::libs::cimg::show(arrange_filters(wt, fa,fb, ds.image_size,ds.channels), "weights");
    return 0;
}
