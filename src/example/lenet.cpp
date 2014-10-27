#include <cuvnet/ops.hpp>
#include <cuvnet/models/models.hpp>
#include <cuvnet/models/conv_layer.hpp>
#include <cuvnet/models/mlp.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/tools/simple_dataset_learner.hpp>
#include <cuvnet/tools/logging.hpp>
#include <google/protobuf/text_format.h>

namespace cuvnet{namespace models{
struct lenet : public metamodel<model>{
    typedef boost::shared_ptr<Op> op_ptr;
    typedef boost::shared_ptr<ParameterInput> input_ptr;
    std::vector<boost::shared_ptr<model> > m_layers;
    lenet(){
        m_input = input(cuv::extents[64][28*28*1], "X");
        m_teacher = input(cuv::extents[64][10], "Y");
        model::m_inputs = {m_input.get(), m_teacher.get()};
        auto cl0 = boost::make_shared<conv_layer>( 
                reorder_for_conv(reshape(m_input, cuv::extents[-1][1][28][28], false)), 
                5, 20,
                conv_layer_opts().with_bias(false, 0.0)
                //.nonlinearity(cuvnet::tanh)
                .pool(2));
        auto cl1 = boost::make_shared<conv_layer>(
                cl0->m_output, 5, 50,
                conv_layer_opts().with_bias(false, 0.0)
                //.nonlinearity(cuvnet::tanh)
                .pool(2));
        auto fc0 = boost::make_shared<mlp_layer>(
                flatten(reorder_from_conv(cl1->m_output), 2, false), 500,
                mlp_layer_opts().with_bias(false,0.0)
                .rectified_linear(false).dropout(true, false)
                );
        auto lr = boost::make_shared<logistic_regression>(
                fc0->m_output, m_teacher);

        register_submodel(*cl0);
        register_submodel(*cl1);
        register_submodel(*fc0);
        register_submodel(*lr);
        m_layers.push_back(cl0);
        m_layers.push_back(cl1);
        m_layers.push_back(fc0);
        m_layers.push_back(lr);
        m_cerr = lr->error();
        m_loss = lr->loss();
    }
    virtual op_ptr loss()const override{ return m_loss; }
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

    cuvnet::models::lenet model;
    model.reset_params();

    cuvnet::simple_dataset_learner sdl;
    sdl.init("mnist", 1, 10000./60000.);
    cuvnet::msg::Fit cfg;
    cfg.mutable_monitor()->set_verbose(true);
    cfg.mutable_monitor()->set_every(781);

    cuvnet::msg::GradientDescent& gd = *cfg.mutable_gd();
    gd.set_learnrate(0.01);
    gd.set_l2decay(0.0005);
    gd.set_batch_size(64);
    gd.MutableExtension(cuvnet::msg::momentum_ext)->set_momentum(0.9);
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
