#include <boost/test/unit_test.hpp>
//#include <boost/property_tree/ptree.hpp>
//#include <boost/property_tree/json_parser.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/learner2.hpp>
#include <cuvnet/models/mlp.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <cuvnet/models/inception.hpp>

#include <cuvnet/derivative_test.hpp>

BOOST_CLASS_EXPORT(cuvnet::models::linear_regression);

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

    BOOST_AUTO_TEST_CASE(mlp_derivative){
        using namespace cuvnet;
        boost::shared_ptr<ParameterInput> inp = input(cuv::extents[10][15]);
        boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[10][2]);
        tgt->set_derivable(false);
        std::vector<unsigned int> hls(2, 4);
        cuvnet::models::mlp_classifier mlp(inp, tgt, hls);
        derivative_testing::derivative_tester(*mlp.loss()).epsilon(0.001).precision(0.003).test();
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
    
    msg::Fit pt;
    pt.mutable_gd()->set_learnrate(0.1);
    pt.mutable_gd()->set_batch_size(bs);
    msg::LinearSchedule lrsched;
    lrsched.set_final(1e-5);
    pt.mutable_gd()->mutable_stopping_criteria()->set_max_epochs(5);
    *pt.mutable_gd()->MutableExtension(msg::linear_learnrate_schedule) = lrsched;
    msg::EarlyStopper& es = *(pt.mutable_gd()->mutable_stopping_criteria()->mutable_es());
    es.set_watch("cerr");
    pt.mutable_monitor()->set_verbose(true);

    learner2 lrn;
    msg::FitResult result1 = lrn.fit(lr, pt);
    msg::FitResult result2 = lrn.continue_learning_until_previous_loss_reached(lr, pt, result1);
    BOOST_CHECK_GT( result1.early_stopper().best_validation_loss(), result2.cerr());
}

BOOST_AUTO_TEST_CASE(learn){
    using namespace cuvnet;
    int bs = 10;
    boost::shared_ptr<ParameterInput> inp = input(cuv::extents[bs][15]);
    boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[bs][2]);
    cuv::fill_rnd_uniform(inp->data());
    cuv::fill_rnd_uniform(tgt->data());
    cuvnet::models::linear_regression lr(inp, tgt);
    
    msg::Fit pt;
    pt.mutable_gd()->set_learnrate(0.1);
    pt.mutable_gd()->set_batch_size(bs);
    msg::LinearSchedule lrsched;
    lrsched.set_final(1e-5);
    pt.mutable_gd()->mutable_stopping_criteria()->set_max_epochs(5);
    *pt.mutable_gd()->MutableExtension(msg::linear_learnrate_schedule) = lrsched;
    msg::EarlyStopper& es = *(pt.mutable_gd()->mutable_stopping_criteria()->mutable_es());
    es.set_watch("loss");
    pt.mutable_monitor()->set_verbose(true);

    learner2 lrn;
    msg::FitResult result1 = lrn.fit(lr, pt);
    msg::FitResult result2 = lrn.continue_learning_until_previous_loss_reached(lr, pt, result1);
    BOOST_CHECK_GE( result1.early_stopper().best_validation_loss(), result2.loss());
}

BOOST_AUTO_TEST_CASE(learn_multistage){
    using namespace cuvnet;
    int bs = 10;
    boost::shared_ptr<ParameterInput> inp = input(cuv::extents[bs][15], "X");
    boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[bs][2], "Y");
    cuv::fill_rnd_uniform(inp->data());
    cuv::fill_rnd_uniform(tgt->data());
    multistage_testmodel mstm(inp, tgt);
    mstm.reset_params();
    
    msg::Fit pt;
    pt.mutable_gd()->set_learnrate(0.1);
    pt.mutable_gd()->set_batch_size(bs);
    msg::LinearSchedule lrsched;
    lrsched.set_final(1e-5);
    pt.mutable_gd()->mutable_stopping_criteria()->set_max_epochs(5);
    *pt.mutable_gd()->MutableExtension(msg::linear_learnrate_schedule) = lrsched;
    msg::EarlyStopper& es = *(pt.mutable_gd()->mutable_stopping_criteria()->mutable_es());
    es.set_watch("loss");
    pt.mutable_monitor()->set_verbose(true);
    msg::MultiStageFit msf;
    msf.set_switch_stage_with_outputs(true);
    *pt.MutableExtension(msg::multistage_ext) = msf;

    multistage_learner lrn;
    msg::FitResult result1 = lrn.fit(mstm, pt);
    BOOST_REQUIRE(result1.stage_size() == 2);
    int epochs0 = result1.stage(0).result_epoch();
    int epochs1 = result1.stage(1).result_epoch();
    BOOST_CHECK_GT(epochs0, 0);
    BOOST_CHECK_GT(epochs1, 0);
}

BOOST_AUTO_TEST_CASE(xval_learn){
    using namespace cuvnet;
    int bs = 10;
    boost::shared_ptr<ParameterInput> inp = input(cuv::extents[bs][15]);
    boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[bs][2]);
    cuv::fill_rnd_uniform(inp->data());
    cuv::fill_rnd_uniform(tgt->data());
    
    boost::shared_ptr<cuvnet::models::linear_regression> lr
    	= boost::make_shared<cuvnet::models::linear_regression>(inp, tgt);

    //ptree pt;
    //pt.put("fit.gd.learnrate", 0.1f);
    //pt.put("fit.batchsize", bs);
    //pt.put("fit.stages", "finetuning");
    //pt.put("fit.gd.max_epochs", 5);
    //pt.put("fit.path", ".");
    //pt.put("fit.learnrate_schedule.type", "linear");
    //pt.put("fit.learnrate_schedule.final", 1e-5);
    //pt.put("fit.early_stopper.active", true);
    //pt.put("fit.early_stopper.watch", "loss"); // or cerr
    //pt.put("fit.monitor.verbose", true);

    msg::Fit pt;
    pt.mutable_gd()->set_learnrate(0.1);
    pt.mutable_gd()->set_batch_size(bs);
    msg::LinearSchedule lrsched;
    lrsched.set_final(1e-5);
    pt.mutable_gd()->mutable_stopping_criteria()->set_max_epochs(5);
    *pt.mutable_gd()->MutableExtension(msg::linear_learnrate_schedule) = lrsched;
    msg::EarlyStopper& es = *(pt.mutable_gd()->mutable_stopping_criteria()->mutable_es());
    es.set_watch("loss");
    pt.mutable_monitor()->set_verbose(true);
    msg::XVal cfg;
    *cfg.mutable_fit() = pt;
    cfg.mutable_predict()->set_batch_size(bs);

    crossvalidator2 cval;
    learner2 lrn;
    msg::XValResult result1 = cval.fit(lrn, lr, cfg);
    //ptree result2 = lrn.continue_learning_until_previous_loss_reached(lr, pt.get_child("fit"), result1.get_child("xval.best_fold"));
    //BOOST_CHECK_GT(
    //        result1.get<float>("xval.best_fold.gd.early_stopper.best_perf"),
    //        result2.get<float>("loss"));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( t_inception )
    BOOST_AUTO_TEST_CASE(inception_copy){
        using namespace cuvnet;
        boost::shared_ptr<ParameterInput> inp = input(cuv::extents[4][4][4][4], "img");
        std::vector<boost::tuple<int,int,int> > m{
            {-1, 3, 3},  // max-pooling -> 3 maps
            {5, 3, 16},   // 3 maps, then 5x5 filter to 16 maps
            {3, 1, 16},   // 1 map, then 3x3 filter to 16 maps
        };
        cuvnet::models::inception_layer inc1(inp, m, "il1", true);
        cuv::initialize_mersenne_twister_seeds(42);
        inc1.reset_params();
        std::vector<cuv::tensor<float, cuv::host_memory_space> > res_copy
            = derivative_testing::all_outcomes(inc1.m_output);

        cuvnet::models::inception_layer inc2(inp, m, "il2", false);
        cuv::initialize_mersenne_twister_seeds(42);
        inc2.reset_params();
        std::vector<cuv::tensor<float, cuv::host_memory_space> > res_nocopy
            = derivative_testing::all_outcomes(inc2.m_output);

        BOOST_CHECK(res_copy.size() == res_nocopy.size());
        for(unsigned int i=0; i<res_copy.size(); i++){
            double diff = cuv::norm1(res_copy[i] - res_nocopy[i]);
            BOOST_CHECK_LT(diff, 0.01);
        }
    }
    BOOST_AUTO_TEST_CASE(inception_derivative){
        using namespace cuvnet;
        boost::shared_ptr<ParameterInput> inp = input(cuv::extents[4][4][4][4], "img");
        std::vector<boost::tuple<int,int,int> > m{
            {-1, 3, 3},  // max-pooling -> 3 maps
            {5, 3, 16},   // 3 maps, then 5x5 filter to 16 maps
            {3, 1, 16},   // 1 map, then 3x3 filter to 16 maps
        };
        cuvnet::models::inception_layer inc(inp, m);
        derivative_testing::derivative_tester(*inc.m_output).precision(0.05).values(0.0, 4.0).spread_values(true, "^img$").full_jacobian(true).only_param(".*").verbose(true).test();
    }
BOOST_AUTO_TEST_SUITE_END()
