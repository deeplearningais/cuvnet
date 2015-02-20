#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/gradient_descent.hpp>


/**
 * @file logistic_regression.cpp
 *
 * This is an example for multinomial logistic regression.
 *
 * It is slightly more complicated than `linear_regression.cpp`,
 * since the loss we're minimizing (logistic) and the loss 
 * for evaluation (zero-one) are different.
 *
 * The zero-one loss is determined from an intermediate result 
 * of the logistic loss, which needs to be remembered to do so.
 *
 */

typedef boost::shared_ptr<cuvnet::Op> op_ptr;
typedef boost::shared_ptr<cuvnet::ParameterInput> input_ptr;

int main(int argc, char **argv){
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;

    int n_examples=1000, n_in_dim=20, n_out_dim=10;

    // 1. create the loss function
    op_ptr loss, closs;
    input_ptr X,Y,W;
    {
        using namespace cuvnet;
        X = input(cuv::extents[n_examples][n_in_dim],"X");
        Y = input(cuv::extents[n_examples],"Y");
        W = input(cuv::extents[n_in_dim][n_out_dim],"W");

        op_ptr estimator = prod(X,W);
        loss = mean(
                multinomial_logistic_loss2(
                    estimator, Y));
        closs = result(loss, 2);
    }

    // 2. create some (dummy) dataset
    {
        using namespace cuv;
        fill_rnd_uniform(X->data());
        fill_rnd_uniform(W->data());
        tensor<float,cuv::dev_memory_space> Y_(cuv::extents[n_examples][n_out_dim]);
        prod(Y_, X->data(), W->data());

        // Y is a one-out-of-n coding for n_out_dim classes
        cuv::reduce_to_col(Y->data(), Y_, cuv::RF_ARGMAX);
        
        // forget about W again
        W->data() = 0.f;
    }

    // 3. create a monitor for the loss
    // Note that here, we must tell the monitor to evaluate
    // the class loss before its statistics are available,
    // therefore we use the WP_FUNC_SCALAR_EPOCH_STATS 
    // watchpoint type.
    cuvnet::monitor mon(true);
    mon.add(cuvnet::monitor::WP_SCALAR_EPOCH_STATS, loss, "loss");
    mon.add(cuvnet::monitor::WP_SCALAR_EPOCH_STATS, closs, "class loss");

    // 4. recover W with logistic regression
    {
        std::vector<cuvnet::Op*> params(1,W.get());
        cuvnet::gradient_descent gd(loss,0,params,1.5f);
        gd.get_swiper().dump("logistic_regression.dot", false);
        mon.register_gd(gd);
        gd.batch_learning(1000);
    }

    std::cout << "\nFinal multinomial logistic loss: "<<mon.mean("loss")<<std::endl;
    std::cout << "\nFinal classification error: "<<mon.mean("class loss")<<std::endl;
}
