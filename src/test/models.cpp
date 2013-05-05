#include <boost/test/unit_test.hpp>
//#include <boost/property_tree/ptree.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/learner2.hpp>
#include <cuvnet/models/mlp.hpp>
#include <cuvnet/models/linear_regression.hpp>




BOOST_AUTO_TEST_SUITE( t_mlp )
    BOOST_AUTO_TEST_CASE(initialize){
        using namespace cuvnet;
        boost::shared_ptr<ParameterInput> inp = input(cuv::extents[10][15]);
        boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[10][2]);
        std::vector<unsigned int> hls(1, 4);
        cuvnet::models::mlp_classifier mlp(inp, tgt, hls);
        gradient_descent gd(mlp.loss(), 0, mlp.get_params(), 0.01f);
        gd.batch_learning(1);
    }

BOOST_AUTO_TEST_CASE(learn){
    using namespace cuvnet;
    int bs = 10;
    boost::shared_ptr<ParameterInput> inp = input(cuv::extents[bs][15]);
    boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[bs][2]);
    cuv::fill_rnd_uniform(inp->data());
    cuv::fill_rnd_uniform(tgt->data());
    cuvnet::models::linear_regression lr(inp, tgt);
    
    using boost::property_tree::ptree;
    ptree pt;
    pt.put("fit.gd.learnrate", 0.1f);
    pt.put("fit.gd.batchsize", bs);
    pt.put("fit.gd.max_epochs", 5);
    pt.put("fit.learnrate_schedule.type", "linear");
    pt.put("fit.learnrate_schedule.final", 1e-5);
    pt.put("fit.early_stopper.active", true);
    pt.put("fit.early_stopper.watch", "loss"); // or cerr
    pt.put("fit.monitor.verbose", true);

    learner2 lrn;
    ptree result1 = lrn.fit(lr, pt.get_child("fit"));
    ptree result2 = lrn.continue_learning_until_previous_loss_reached(lr, pt.get_child("fit"), result1);
    BOOST_CHECK_LT(
            result1.get<float>("early_stopper.best_perf"),
            result2.get<float>("loss"));
}
BOOST_AUTO_TEST_SUITE_END()
