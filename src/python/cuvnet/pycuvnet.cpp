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

enum allo_t{
    AT_DEFAULT,
    AT_POOLED_CUDA,
    AT_NAN
};

void init_cuvnet(unsigned int dev, unsigned int seed, allo_t alloc, unsigned int loglevel, std::string logfile){
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

    def("initialize", &init_cuvnet, (args("dev")=0, args("seed")=42, 
                args("alloc")=AT_POOLED_CUDA, 
                args("loglevel")=0, args("logfile")=std::string("log.xml")));
    export_module();

    class_<models::model, boost::shared_ptr<models::model> >("model", no_init)
        .def_readonly("loss", &models::model::loss)
        .def_readonly("error", &models::model::error)
        .def_readonly("inputs", &models::model::get_inputs)
        .def("get_params", &models::model::get_params)
        ;

    class_<models::metamodel<models::multistage_model>, bases<models::model>, 
        boost::shared_ptr<models::metamodel<models::multistage_model> > >("multistage_metamodel", no_init)
        .def_readonly("loss", &models::model::loss)
        .def_readonly("error", &models::model::error)
        .def("get_params", &models::model::get_params)
        ;

    class_<models::mlp_layer, bases<models::model>, boost::shared_ptr<models::mlp_layer> >("mlp_layer", no_init)
        .def_readonly("output", &models::mlp_layer::m_output)
        .def_readonly("linear_output", &models::mlp_layer::m_linear_output)
        .def_readonly("weights", &models::mlp_layer::m_W)
        .def_readonly("bias", &models::mlp_layer::m_bias)
        ;

    class_<models::conv_layer, boost::shared_ptr<models::conv_layer> >("conv_layer", no_init)
        .def_readonly("input", &models::conv_layer::m_input)
        .def_readonly("output", &models::conv_layer::m_output)
        .def_readonly("weights", &models::conv_layer::m_weights)
        .def_readonly("bias", &models::conv_layer::m_bias)
        ;

    class_<models::logistic_regression, boost::shared_ptr<models::logistic_regression> >("logistic_regression", no_init)
        .def_readonly("estimator", &models::logistic_regression::m_estimator)
        .def_readonly("W", &models::logistic_regression::m_W)
        .def_readonly("bias", &models::logistic_regression::m_bias)
        .def_readonly("loss", &models::logistic_regression::m_loss)
        .def_readonly("y", &models::logistic_regression::m_Y)
        ;


}
