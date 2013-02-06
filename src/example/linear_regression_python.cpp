#include <boost/python.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <cuvnet/tools/python_helper.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/gradient_descent.hpp>

/**
 * @file linear_regression_python.cpp
 *
 * This file demonstrates how to export a model to python.
 *
 */

typedef boost::shared_ptr<cuvnet::Op> op_ptr;
typedef boost::shared_ptr<cuvnet::ParameterInput> input_ptr;

struct linear_regression{
    op_ptr m_loss;
    input_ptr m_X, m_Y, m_W;
    op_ptr m_output;
    linear_regression(int n_examples, int n_in_dim, int n_out_dim){
        using namespace cuvnet;
        m_X = input(cuv::extents[n_examples][n_in_dim], "X");
        m_Y = input(cuv::extents[n_examples][n_out_dim], "Y");
        m_W = input(cuv::extents[n_in_dim][n_out_dim], "W");
        m_output = prod(m_X,m_W);

        m_loss = mean(
                sum_to_vec(
                    pow(m_output - m_Y, 2.f), 0));
    }   
};


BOOST_PYTHON_MODULE(pylinreg)
{
    cuvnet::export_module();

    using namespace boost::python;
    class_<linear_regression>("linear_regression", init<int,int,int>())
        .def_readonly("output", &linear_regression::m_output)
        .def_readonly("X", &linear_regression::m_X)
        .def_readonly("Y", &linear_regression::m_Y)
        .def_readonly("W", &linear_regression::m_W)
        .def_readonly("loss", &linear_regression::m_loss);
}
