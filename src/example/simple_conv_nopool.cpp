
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <tools/gradient_descent.hpp>
#include <datasets/letters_inpainting.hpp>

#include <datasets/randomizer.hpp>
#include <tools/preprocess.hpp>
#include <tools/monitor.hpp>
#include <tools/logging.hpp>
#include <tools/python_helper.hpp>

#include <cuvnet/models/conv_nopool.hpp>

using namespace boost::assign;
using namespace cuvnet;
namespace ll = boost::lambda;

int main(int argc, char **argv)
{
    // initialize cuv library   
    cuv::initCUDA(0);
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;

    // load the dataset
    letters_inpainting ds("/home/VI/staff/amueller/datasets/letters/");
    //randomizer().transform(ds.train_data, ds.train_labels);
   

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input and target correctly
    // \c load_batch will put values in it.
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[10][1][200][200],"input"));
    boost::shared_ptr<ParameterInput> target(
            new ParameterInput(cuv::extents[10][1][200][200],"target"));

    // creates the LeNet
    conv_nopool net(5, 16, 5);
    net.init(input, target);


    std::vector<Op*> params = net.params();

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, net.get_loss(),        "total_loss");
    mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, net.classification_error(),        "classification_error");

    // copy training data and labels to the device, and converts train_labels from int to float
    matrix train_data = ds.train_data;
    matrix train_labels(ds.train_labels);
    input->data() = train_data;
    target->data() = train_labels;
    
    std::cout << std::endl << " Training phase: " << std::endl;
    {
        // create a \c gradient_descent object that derives the logistic loss
        // w.r.t. \c params and has learning rate 0.1f
        rprop_gradient_descent gd(net.get_loss(),0,params,0.1f);
        
        // register the monitor so that it receives learning events
        mon.register_gd(gd);

        gd.batch_learning(100, 10*60); // 10 minutes maximum
    }
    initialize_python();
    export_ops();
    export_op("loss", net.get_loss());
    embed_python();
    std::cout << std::endl << " Test phase: " << std::endl;

    // evaluates test data. We use minibatch learning with learning rate zero and only one epoch.
    {
        matrix test_data = ds.test_data;
        matrix test_labels(ds.test_labels);
        gradient_descent gd(net.get_loss(),0,params,0.f);
        mon.register_gd(gd);

        gd.batch_learning(1, 100);
    }

    std::cout <<  std::endl;

     
    return 0;
}
