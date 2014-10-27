#include <cuvnet/ops.hpp>
#include <cuvnet/models/models.hpp>
#include <cuvnet/models/conv_layer.hpp>
#include <cuvnet/models/mlp.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/tools/simple_dataset_learner.hpp>
#include <cuvnet/tools/logging.hpp>
#include <google/protobuf/text_format.h>

namespace cuvnet{namespace models{
struct svm : public metamodel<model>{
    typedef boost::shared_ptr<Op> op_ptr;
    typedef boost::shared_ptr<ParameterInput> input_ptr;
    std::vector<boost::shared_ptr<model> > m_layers;
    svm(){
        m_input = input(cuv::extents[64][28*28*1], "X");
        m_teacher = input(cuv::extents[64][10], "Y");
        model::m_inputs = {m_input.get(), m_teacher.get()};

        auto fc0 = boost::make_shared<mlp_layer>(m_input, 10,
                mlp_layer_opts().with_bias(true,0.0));
        
        auto estimator = fc0->m_output;
        register_submodel(*fc0);
        m_layers.push_back(fc0);

        m_loss = cuvnet::rectified_linear(-2.f * ((m_teacher + -.5f) * estimator) + 1.f, false);
        m_cerr = cuvnet::classification_loss(m_teacher, estimator, 1);
    }
    virtual op_ptr loss()const override{ return m_loss; }
    virtual op_ptr error()const override{ return m_cerr; }
    private:
    op_ptr m_loss, m_cerr;
    input_ptr m_input, m_teacher;
};
}}

int
main(int , char **)
{
    cuv::initCUDA(1);
    cuv::initialize_mersenne_twister_seeds(42);
    cuvnet::Logger log;

    cuvnet::models::svm model;
    model.reset_params();

    cuvnet::simple_dataset_learner sdl;
    sdl.init("mnist", 1, 10000./60000.);
    cuvnet::msg::Fit cfg;
    cfg.mutable_monitor()->set_verbose(true);
    cfg.mutable_monitor()->set_every(781);

    cuvnet::msg::GradientDescent& gd = *cfg.mutable_gd();
    gd.set_learnrate(0.01);
    gd.set_l2decay(0.00005);
    gd.set_batch_size(64);
    //gd.MutableExtension(cuvnet::msg::momentum_ext)->set_momentum(0.9);
    gd.MutableExtension(cuvnet::msg::exponential_learnrate_schedule)->set_duration(10000);
    gd.mutable_stopping_criteria()->set_max_epochs(100000/(60000/64.));
    gd.mutable_stopping_criteria()->mutable_es()->set_active(true);
    gd.mutable_stopping_criteria()->mutable_es()->set_every(1);
    gd.mutable_stopping_criteria()->mutable_es()->set_batch_size(100);

    sdl.switch_dataset(cuvnet::CM_TRAIN);
    sdl.fit(model, cfg);

    // now predict on the test set
    cuvnet::msg::Predict pcfg;
    pcfg.mutable_monitor()->set_verbose(true);
    pcfg.set_batch_size(100);
    sdl.switch_dataset(cuvnet::CM_TEST);
    auto res = sdl.predict(model, pcfg);
    std::string s;
    google::protobuf::TextFormat::PrintToString(res, &s);
    std::cout << "\n\nresult: " << s << std::endl;
    return 0;
}
