#include <boost/test/unit_test.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/models/mlp.hpp>
#include <cuvnet/tools/gradient_descent.hpp>


BOOST_AUTO_TEST_SUITE( t_mlp )
BOOST_AUTO_TEST_CASE(initialize){
    using namespace cuvnet;
    boost::shared_ptr<ParameterInput> inp = input(cuv::extents[10][15]);
    boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[10][2]);
    std::vector<unsigned int> hls(1, 4);
    cuvnet::models::mlp mlp(inp, tgt, hls);
    gradient_descent gd(mlp.loss(), 0, mlp.get_params(), 0.01f);
    gd.batch_learning(1);
}
BOOST_AUTO_TEST_SUITE_END()
