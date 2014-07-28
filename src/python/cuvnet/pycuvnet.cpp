#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/python.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cuv/basics/allocators.hpp>
#include <cuvnet/tools/python_helper.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/models/conv_layer.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/mlp.hpp>
#include <cuvnet/ops.hpp>


namespace bar = boost::archive;
using namespace cuvnet;

typedef boost::property_tree::ptree ptree;
typedef boost::shared_ptr<Op> op_ptr;

enum allo_t{
    AT_DEFAULT,
    AT_POOLED_CUDA,
    AT_NAN
};

void init_cuvnet(int dev, unsigned int seed, allo_t alloc, unsigned int loglevel, std::string logfile){
    cuv::initCUDA(dev);
    if(alloc == AT_DEFAULT)
        cuvnet::cow_ptr<matrix>::s_allocator.reset(
                new cuv::default_allocator());
    else if(alloc == AT_POOLED_CUDA){
        cuvnet::cow_ptr<matrix>::s_allocator.reset(
                new cuv::pooled_cuda_allocator());
    }else if(alloc == AT_NAN){
        cuvAssert(false);
        //cuvnet::cow_ptr<matrix>::s_allocator.reset(
                //new cuv::nan_pooled_cuda_allocator());
    }
    cuv::initialize_mersenne_twister_seeds(seed);
    Logger(logfile, loglevel);
}

BOOST_PYTHON_MODULE(_pycuvnet)
{
    using namespace boost::python;
    enum_<allo_t>("allo_t")
        .value("DEFAULT", AT_DEFAULT)
        .value("POOLED_CUDA", AT_POOLED_CUDA)
        .value("NAN", AT_NAN);

    enum_<cuv::alex_conv::pool_type>("pool_type")
        .value("AVG", cuv::alex_conv::PT_AVG)
        .value("MAX", cuv::alex_conv::PT_MAX)
        ;

    def("initialize", &init_cuvnet, (args("dev")=0, args("seed")=42, 
                args("alloc")=AT_POOLED_CUDA, 
                args("loglevel")=0, args("logfile")=std::string("log.xml")));
    export_module();

    class_<models::model, boost::shared_ptr<models::model> >("model", no_init)
        .def_readonly("loss", &models::model::loss)
        .def_readonly("error", &models::model::error)
        .def_readonly("inputs", &models::model::get_inputs)
        .def("set_predict_mode", &models::model::set_predict_mode)
        .def("get_params", &models::model::get_params)
        .def("reset_params", &models::model::reset_params)
        .def("set_batchsize", &models::model::set_batchsize)
        ;

    class_<models::multistage_model, boost::shared_ptr<models::multistage_model>, bases<models::model> >("multistage_model", no_init)
        ;

    typedef models::metamodel<models::multistage_model> mmetamodel;
    class_<mmetamodel, bases<models::multistage_model>, boost::shared_ptr<mmetamodel> >("multistage_metamodel", init<>())
        .def_readonly("loss", &models::model::loss)
        .def_readonly("error", &models::model::error)
        .def("get_params", &models::model::get_params)
        .def("register_submodel", &mmetamodel::register_submodel)
        .def("deregister_submodel", &mmetamodel::deregister_submodel)
        .def("clear_submodels", &mmetamodel::clear_submodels)
        ;

    typedef models::metamodel<models::model> metamodel;
    class_<metamodel, bases<models::model>, boost::shared_ptr<metamodel> >("metamodel", init<>())
        .def_readonly("loss", &models::model::loss)
        .def_readonly("error", &models::model::error)
        .def("get_params", &models::model::get_params)
        .def("register_submodel", &metamodel::register_submodel)
        .def("deregister_submodel", &metamodel::deregister_submodel)
        .def("clear_submodels", &metamodel::clear_submodels)
        ;

    class_<models::logistic_regression, bases<models::model>, boost::shared_ptr<models::logistic_regression> >("logistic_regression", init<op_ptr, op_ptr, bool, bool>())
        .def(init<op_ptr, op_ptr, int, bool>())
        .def_readonly("estimator", &models::logistic_regression::m_estimator)
        .def_readonly("W", &models::logistic_regression::m_W)
        .def_readonly("bias", &models::logistic_regression::m_bias)
        .def_readonly("loss", &models::logistic_regression::m_loss)
        .def_readonly("cerr", &models::logistic_regression::m_classloss)
        .def_readonly("y", &models::logistic_regression::m_Y)
        .def_readonly("x", &models::logistic_regression::m_X)
        ;

    class_<models::linear_regression, bases<models::model>, boost::shared_ptr<models::linear_regression> >("linear_regression", init<op_ptr, op_ptr, bool, bool>())
        .def_readonly("W", &models::linear_regression::m_W)
        .def_readonly("bias", &models::linear_regression::m_bias)
        .def_readonly("loss", &models::linear_regression::m_loss)
        ;


    class_<models::mlp_layer_opts>("mlp_layer_opts")
        .def("copy", &models::mlp_layer_opts::copy)
        .def("verbose", &models::mlp_layer_opts::verbose, return_internal_reference<>())
        .def("rectified_linear", &models::mlp_layer_opts::rectified_linear, return_internal_reference<>())
        .def("tanh", &models::mlp_layer_opts::tanh, return_internal_reference<>())
        .def("logistic", &models::mlp_layer_opts::logistic, return_internal_reference<>())
        .def("with_bias", &models::mlp_layer_opts::with_bias, (args("b")=true, args("defaultval")=0.f), return_internal_reference<>())
        .def("maxout", &models::mlp_layer_opts::maxout, (args("n")), return_internal_reference<>())
        .def("dropout", &models::mlp_layer_opts::dropout, (args("b")=true), return_internal_reference<>())
        .def("learnrate_factor", &models::mlp_layer_opts::learnrate_factor, (args("fW"), args("fB")=-1.f), return_internal_reference<>())
        .def("group", &models::mlp_layer_opts::group, (args("name")="", args("unique")=true), return_internal_reference<>())
        .def("weight_init_std", &models::mlp_layer_opts::weight_init_std, (args("std")), return_internal_reference<>())
        ;


    class_<models::conv_layer_opts>("conv_layer_opts")
        .def("copy", &models::conv_layer_opts::copy)
        .def("rectified_linear", &models::conv_layer_opts::linear, return_internal_reference<>())
        .def("linear", &models::conv_layer_opts::linear, return_internal_reference<>())
        .def("learnrate_factor", &models::conv_layer_opts::learnrate_factor, (args("fW"), args("fB")=-1.f), return_internal_reference<>())
        .def("random_sparse", &models::conv_layer_opts::random_sparse, return_internal_reference<>())
        .def("verbose", &models::conv_layer_opts::verbose, return_internal_reference<>())
        .def("pool", &models::conv_layer_opts::pool, (args("pool_size"), args("pool_stride")=-1, args("pt")=cuv::alex_conv::PT_MAX), return_internal_reference<>())
        .def("contrast_norm", &models::conv_layer_opts::contrast_norm, (args("n"), args("alpha")=0.001f, args("beta")=0.5f), return_internal_reference<>())
        .def("response_norm", &models::conv_layer_opts::response_norm, (args("n"), args("alpha")=0.5f, args("beta")=0.5f), return_internal_reference<>())
        .def("padding", &models::conv_layer_opts::padding, (args("i")=-1), return_internal_reference<>())
        .def("symmetric_padding", &models::conv_layer_opts::symmetric_padding, (args("i")=-1), return_internal_reference<>())
        .def("stride", &models::conv_layer_opts::stride, (args("i")=-1), return_internal_reference<>())
        .def("n_groups", &models::conv_layer_opts::n_groups, (args("i")), return_internal_reference<>())
        .def("n_filter_channels", &models::conv_layer_opts::n_filter_channels, (args("i")), return_internal_reference<>())
        .def("partial_sum", &models::conv_layer_opts::partial_sum, (args("i")), return_internal_reference<>())
        .def("with_bias", &models::conv_layer_opts::with_bias, (args("b")=true, args("defaultval")=0.f), return_internal_reference<>())
        .def("maxout", &models::conv_layer_opts::maxout, (args("n")), return_internal_reference<>())
        .def("dropout", &models::conv_layer_opts::dropout, (args("rate")=0.5f), return_internal_reference<>())
        .def("group", &models::conv_layer_opts::group, (args("name")="", args("unique")=true), return_internal_reference<>())
        .def("weight_init_std", &models::conv_layer_opts::weight_default_std, (args("std")), return_internal_reference<>())
        ;


    class_<models::mlp_layer, bases<models::model>, boost::shared_ptr<models::mlp_layer>, bases<models::model> >("mlp_layer", init<op_ptr, unsigned int, models::mlp_layer_opts>())
        .def_readonly("output", &models::mlp_layer::m_output)
        .def_readonly("linear_output", &models::mlp_layer::m_linear_output)
        .def_readonly("weights", &models::mlp_layer::m_W)
        .def_readonly("bias", &models::mlp_layer::m_bias)
        ;

    class_<models::conv_layer, boost::shared_ptr<models::conv_layer>, bases<models::model> >("conv_layer", init<op_ptr,int,int,const models::conv_layer_opts&>())
        .def_readonly("input", &models::conv_layer::m_input)
        .def_readonly("output", &models::conv_layer::m_output)
        .def_readonly("weights", &models::conv_layer::m_weights)
        .def_readonly("bias", &models::conv_layer::m_bias)
        ;
}
