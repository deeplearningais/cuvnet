#include <boost/test/unit_test.hpp>
//#include <boost/property_tree/ptree.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/monitor.hpp>
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

struct tmplogreg : public cuvnet::models::logistic_regression{
    tmplogreg(boost::shared_ptr<cuvnet::ParameterInput> inp, boost::shared_ptr<cuvnet::ParameterInput> tgt)
        : cuvnet::models::logistic_regression(inp,tgt,false){}

    void register_watches(cuvnet::monitor& mon, const std::string& stage = ""){
        mon.add(cuvnet::monitor::WP_SINK, m_estimator, "output");
    }
};

struct multistage_testmodel : public cuvnet::models::multistage_model {
    typedef boost::shared_ptr<cuvnet::Op> op_ptr;
    typedef boost::shared_ptr<cuvnet::ParameterInput> input_ptr;
    input_ptr m_weights0;
    op_ptr m_hidden;
    op_ptr m_loss0;
    boost::shared_ptr<cuvnet::models::linear_regression> m_linreg;

    multistage_testmodel(input_ptr inp, input_ptr tgt)
        : multistage_model(2)
    {
        using namespace cuvnet;
        determine_shapes(*inp);
        int fdim = inp->result()->shape[1];
        int hdim = 5;
        m_weights0 = input(cuv::extents[fdim][hdim], "W0");
        m_hidden = label("hidden", prod(inp, m_weights0));
        m_loss0 = mean(square(inp - prod(m_hidden, m_weights0, 'n', 't')));
        m_linreg.reset(new cuvnet::models::linear_regression(m_hidden, tgt));
    }
    virtual std::vector<cuvnet::Op*> get_outputs(){
        using namespace cuvnet;
        std::vector<Op*> v;
        if(current_stage() == 0)
            v.push_back(m_hidden.get());
        else {
            cuvAssert(false);
        }
        return v;
    }
    op_ptr loss()const{
        if(current_stage() == 0)
            return m_loss0;
        return m_linreg->loss();
    }
    std::vector<cuvnet::Op*> get_params(){
        using namespace cuvnet;
        if(current_stage() == 0)
            return std::vector<Op*> (1, m_weights0.get());
        return m_linreg->get_params();
    }
    void reset_params(){
        cuv::fill_rnd_uniform(m_weights0->data());
        m_weights0->data() *= 0.05f;
        m_weights0->data() -= 0.25f;
        m_linreg->reset_params();
    }
};

BOOST_AUTO_TEST_CASE(nenadbug){
    // this is a regression test for an error in traversing models when a
    // WP_SINK was attached
    using namespace cuvnet;
    int bs = 10;
    boost::shared_ptr<ParameterInput> inp = input(cuv::extents[bs][15]);
    boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[bs][2]);
    cuv::fill_rnd_uniform(inp->data());
    cuv::fill_rnd_uniform(tgt->data());
    tmplogreg lr(inp, tgt);
    
    using boost::property_tree::ptree;
    ptree pt;
    pt.put("fit.gd.learnrate", 0.1f);
    pt.put("fit.batchsize", bs);
    pt.put("fit.gd.max_epochs", 5);
    pt.put("fit.path", ".");
    pt.put("fit.learnrate_schedule.type", "linear");
    pt.put("fit.learnrate_schedule.final", 1e-5);
    pt.put("fit.early_stopper.active", true);
    pt.put("fit.early_stopper.watch", "loss"); // or cerr
    pt.put("fit.monitor.verbose", true);

    learner2 lrn;
    ptree result1 = lrn.fit(lr, pt.get_child("fit"));
    ptree result2 = lrn.continue_learning_until_previous_loss_reached(lr, pt.get_child("fit"), result1);
    BOOST_CHECK_GT(
           result1.get<float>("gd.early_stopper.best_perf"),
           result2.get<float>("cerr"));
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
    pt.put("fit.batchsize", bs);
    pt.put("fit.gd.max_epochs", 5);
    pt.put("fit.path", ".");
    pt.put("fit.learnrate_schedule.type", "linear");
    pt.put("fit.learnrate_schedule.final", 1e-5);
    pt.put("fit.early_stopper.active", true);
    pt.put("fit.early_stopper.watch", "loss"); // or cerr
    pt.put("fit.monitor.verbose", true);

    learner2 lrn;
    ptree result1 = lrn.fit(lr, pt.get_child("fit"));
    ptree result2 = lrn.continue_learning_until_previous_loss_reached(lr, pt.get_child("fit"), result1);
    BOOST_CHECK_GT(
           result1.get<float>("gd.early_stopper.best_perf"),
           result2.get<float>("loss"));
}

BOOST_AUTO_TEST_CASE(learn_multistage){
    using namespace cuvnet;
    int bs = 10;
    boost::shared_ptr<ParameterInput> inp = input(cuv::extents[bs][15], "X");
    boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[bs][2], "Y");
    cuv::fill_rnd_uniform(inp->data());
    cuv::fill_rnd_uniform(tgt->data());
    multistage_testmodel mstm(inp, tgt);
    
    using boost::property_tree::ptree;
    ptree pt;
    pt.put("fit.gd.learnrate", 0.1f);
    pt.put("fit.batchsize", bs);
    pt.put("fit.gd.max_epochs", 5);
    pt.put("fit.path", ".");
    pt.put("fit.learnrate_schedule.type", "linear");
    pt.put("fit.learnrate_schedule.final", 1e-5);
    pt.put("fit.early_stopper.active", true);
    pt.put("fit.early_stopper.watch", "loss"); // or cerr
    pt.put("fit.monitor.verbose", true);
    pt.put("fit.switch_stage_with_outputs", true);

    multistage_learner lrn;
    ptree result1 = lrn.fit(mstm, pt.get_child("fit"));
}

BOOST_AUTO_TEST_CASE(xval_learn){
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
    pt.put("fit.batchsize", bs);
    pt.put("fit.stages", "finetuning");
    pt.put("fit.gd.max_epochs", 5);
    pt.put("fit.path", ".");
    pt.put("fit.learnrate_schedule.type", "linear");
    pt.put("fit.learnrate_schedule.final", 1e-5);
    pt.put("fit.early_stopper.active", true);
    pt.put("fit.early_stopper.watch", "loss"); // or cerr
    pt.put("fit.monitor.verbose", true);

    crossvalidator2 cval(1);
    learner2 lrn;
    ptree result1 = cval.fit(lrn, lr, pt.get_child("fit"));
    //ptree result2 = lrn.continue_learning_until_previous_loss_reached(lr, pt.get_child("fit"), result1.get_child("xval.best_fold"));
    //BOOST_CHECK_GT(
    //        result1.get<float>("xval.best_fold.gd.early_stopper.best_perf"),
    //        result2.get<float>("loss"));
}
BOOST_AUTO_TEST_SUITE_END()
