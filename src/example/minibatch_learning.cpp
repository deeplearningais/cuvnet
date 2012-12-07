#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/gradient_descent.hpp>

/**
 * @file minibatch_learning.cpp
 *
 * This file demonstrates how to do linear regression in an online fashion
 * using an endless dataset.
 *
 * We will create the data ourselves, but of course you can simply drop in 
 * your favorite dataset. Cuvnet does not impose any constraints on
 * how the dataset should be represented. You only need to provide a 
 * function which is called before the forward pass of every minibatch.
 * This function should copy some data into the Input objects.
 */

typedef boost::shared_ptr<cuvnet::Op> op_ptr;
typedef boost::shared_ptr<cuvnet::ParameterInput> input_ptr;

struct data{
    cuv::tensor<float, cuvnet::matrix::memory_space_type> W;
    data(int n_in_dim, int n_out_dim)
        : W(cuv::extents[n_in_dim][n_out_dim]) {
        // create a weight matrix which is unknown to the learner
        // and must be recovered by it.
        fill_rnd_uniform(W);
    }

    void load_batch(input_ptr X, input_ptr Y){
        // generate Y from the underlying distribution
        fill_rnd_uniform(X->data());
        prod(Y->data(), X->data(), W);
        
        // corrupt Y with Gaussian noise
        add_rnd_normal(Y->data(), 0.05f);
    }

    unsigned int n_batches(){ return 10u; }
};

int main(int argc, char **argv){
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;

    int n_batchsize=64, n_in_dim=20, n_out_dim=10;

    // 1. create the loss function
    op_ptr loss;
    input_ptr X,Y,W;
    {
        using namespace cuvnet;
        X = input(cuv::extents[n_batchsize][n_in_dim]);
        Y = input(cuv::extents[n_batchsize][n_out_dim]);
        W = input(cuv::extents[n_in_dim][n_out_dim]);

        loss = mean(
                sum_to_vec(
                    pow(prod(X,W)-Y, 2.f), 0));
    }

    // 2. create some (dummy) dataset
    data ds(n_in_dim, n_out_dim);
    W->data() = 0.f;

    // 3. create a monitor for the loss
    cuvnet::monitor mon(true);
    mon.add(cuvnet::monitor::WP_SCALAR_EPOCH_STATS, loss, "loss");

    // 4. recover W with online linear regression
    {
        std::vector<cuvnet::Op*> params(1,W.get());
        cuvnet::gradient_descent gd(loss,0,params,0.1f);

        // this is the main difference between batch- and minibatch learning:
        // We need to tell gradient_descent how to load batches,
        // and how many batches there are in one `epoch'.
        gd.before_batch.connect(boost::bind(&data::load_batch,&ds,X,Y));
        gd.current_batch_num = boost::bind(&data::n_batches, &ds);
        mon.register_gd(gd);
        gd.minibatch_learning(100);
    }

    std::cout << "\nFinal mean squared error: "<<mon.mean("loss")<<std::endl;
}
