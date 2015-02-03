#include <boost/test/unit_test.hpp>
//#include <boost/property_tree/ptree.hpp>
//#include <boost/property_tree/json_parser.hpp>
#include <cuvnet/ops.hpp>
#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/learner2.hpp>
#include <cuvnet/tools/matwrite.hpp>
#include <cuvnet/models/mlp.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <cuvnet/models/inception.hpp>

#include <cuvnet/derivative_test.hpp>

BOOST_AUTO_TEST_SUITE( t_graddesc )
    BOOST_AUTO_TEST_CASE(initialize){
        using namespace cuvnet;
        boost::shared_ptr<ParameterInput> inp = input(cuv::extents[1][1]);
        boost::shared_ptr<ParameterInput> tgt = input(cuv::extents[1][1]);
        cuvnet::models::linear_regression mlp(inp, tgt);
        inp->data() = 0.0f;
        tgt->data() = 1.0f;
        host_matrix m;
        {
            swiper swipe(*mlp.loss(), 0, {inp.get()});
            swipe.fprop();
            swipe.bprop();
            m = inp->cdelta().copy();
            BOOST_CHECK_GT(cuv::norm1(m), 0.1f); // not trivial
            swipe.fprop();
            swipe.bprop();
            host_matrix m2 = inp->cdelta().copy();
            m *= 2.f;
            BOOST_CHECK_LT(cuv::norm1(m-m2), 0.001f);
            inp->reset_delta();
        }
        {
            gradient_descent gd(mlp.loss(), 0, {inp.get()}, 1.f, 0.0f);
            gd.set_update_every(2);
            gd.current_batch_num = []{ return 2; };
            {
                // check that gradient is really updated after 2 steps
                gd.minibatch_learning(1);
                host_matrix m2 = inp->data().copy();
                BOOST_CHECK_LT(cuv::norm1(m+m2), 0.0001f);  // gradient is negative.
            }
            {
                // check that there is no 'residual' by restarting from 0
                inp->data() = 0.f;
                gd.minibatch_learning(2);
                host_matrix m2 = inp->data().copy();
                BOOST_CHECK_LT(cuv::norm1(m+m2), 0.001f);
            }
        }
        {
            momentum_gradient_descent gd(mlp.loss(), 0, {inp.get()}, 1.f, 0.0f, 0.5f);
            gd.set_update_every(2);
            gd.current_batch_num = []{ return 2; };
            {
                // check that gradient is really updated after 2 steps
                inp->data() = 0.f;
                gd.minibatch_learning(1);
                host_matrix m2 = inp->data().copy();
                BOOST_CHECK_LT(cuv::norm1(m+m2), 0.0001f);  // gradient is negative.
            }
            {
                // this one has a 'residual', it is the momentum kept from the 1st iteration.
                inp->data() = 0.f;
                gd.minibatch_learning(2);
                host_matrix m2 = inp->data().copy();
                BOOST_CHECK_LT(cuv::norm1((1.f + .5f) * m + m2), 0.001f);
            }
        }
    }
BOOST_AUTO_TEST_SUITE_END()
