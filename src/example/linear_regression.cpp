#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/gradient_descent.hpp>

/**
 * @file linear_regression.cpp
 *
 * This is the most basic example, demonstrating the use of cuvnet for a
 * linear_regression with gradient-descent task.
 *
 * The required steps are
 * 1. Generate a loss function
 * 2. Create a dataset (you'll probably want to load real data from a file
 *    instead)
 * 3. Create a monitor for learning
 * 4. Minimize the loss with gradient descent.
 *
 * @note that gradient descent is not necessary for linear regression, this
 * problem can be solved analytically. We use linear regression here since its
 * loss function has a very simple form.
 */

typedef boost::shared_ptr<cuvnet::Op> op_ptr;
typedef boost::shared_ptr<cuvnet::ParameterInput> input_ptr;

int main(int argc, char **argv){
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;

    int n_examples=1000, n_in_dim=20, n_out_dim=10;

    // 1. create the loss function
    op_ptr loss;
    input_ptr X,Y,W;
    {
        using namespace cuvnet;
        X = input(cuv::extents[n_examples][n_in_dim]);
        Y = input(cuv::extents[n_examples][n_out_dim]);
        W = input(cuv::extents[n_in_dim][n_out_dim]);

        loss = mean(
                sum_to_vec(
                    pow(prod(X,W)-Y, 2.f), 0));
    }

    // 2. create some (dummy) dataset
    {
        using namespace cuv;
        fill_rnd_uniform(X->data());
        fill_rnd_uniform(W->data());
        prod(Y->data(), X->data(), W->data());
        
        // forget about W again
        W->data() = 0.f;

        // add gaussian noise to Y
        add_rnd_normal(Y->data(), 0.01f);
    }

    // 3. create a monitor for the loss
    cuvnet::monitor mon(true); // parameter is verbosity
    mon.add(cuvnet::monitor::WP_SCALAR_EPOCH_STATS, loss, "loss");

    // 4. recover W with linear regression
    {
        std::vector<cuvnet::Op*> params(1,W.get());
        cuvnet::gradient_descent gd(loss,0,params,0.1f);

        // the monitor attaches its callbacks to signals of gradient descent,
        // i.e. 
        // - before_epoch resets the epoch statistics,
        // - after_batch records the current values of the watchpoints
        // - after_epoch prints the current stats if the `verbose` is set
        mon.register_gd(gd);
        gd.batch_learning(100);
    }

    std::cout << "\nFinal mean squared error: "<<mon.mean("loss")<<std::endl;
}
