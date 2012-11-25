#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <tools/monitor.hpp>
#include <tools/logging.hpp>
#include <tools/gradient_descent.hpp>


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
        X = input(cuv::extents[n_examples][n_in_dim]);
        Y = input(cuv::extents[n_examples][n_out_dim]);
        W = input(cuv::extents[n_in_dim][n_out_dim]);

        op_ptr estimator = prod(X,W);
        loss = mean(
                multinomial_logistic_loss(
                    estimator, Y, 0));

        // when the forward pass for the loss is calculated, 
        // only the /required/ ops for `loss' are calculated.
        // the `classification_loss' op will not be one of those.
        // Since we don't want to redo all calculations, we 
        // introduce a `sink' here, which captures the intermediate
        // result. If a monitor for classification loss is used,
        // it must be told that the result is not readily available,
        // as a result of the forward pass on the `loss' op;
        // it must be calculated first!
        closs = classification_loss(sink(estimator), Y);
    }

    // 2. create some (dummy) dataset
    {
        using namespace cuv;
        fill_rnd_uniform(X->data());
        fill_rnd_uniform(W->data());
        prod(Y->data(), X->data(), W->data());

        // Y is a one-out-of-n coding for n_out_dim classes
        cuv::tensor<unsigned int> idx(cuv::extents[n_examples]);
        cuv::reduce_to_col(idx, Y->data(), cuv::RF_ARGMAX);
        Y->data() = 0.f;
        for(unsigned int i=0;i < n_examples; i++)
            Y->data()(i,idx(i)) = 1.f;
        
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
    mon.add(cuvnet::monitor::WP_FUNC_SCALAR_EPOCH_STATS, closs, "class loss");

    // 4. recover W with logistic regression
    {
        std::vector<cuvnet::Op*> params(1,W.get());
        cuvnet::gradient_descent gd(loss,0,params,0.1f);
        mon.register_gd(gd);
        gd.batch_learning(100);
    }

    std::cout << "\nFinal mean squared error: "<<mon.mean("loss")<<std::endl;
    std::cout << "\nFinal classification error: "<<mon.mean("class loss")<<std::endl;
}
