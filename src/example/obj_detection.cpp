// vim:ts=4:sw=4:et
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <tools/gradient_descent.hpp>

#include <tools/monitor.hpp>

#include <datasets/voc_detection.hpp>
#include <cuvnet/models/object_detector.hpp>

using namespace boost::assign;
using namespace cuvnet;
namespace ll = boost::lambda;

/**
 * load a batch from the dataset
 */
void load_batch(
        boost::shared_ptr<ParameterInput> input,
        boost::shared_ptr<ParameterInput> ignore,
        boost::shared_ptr<ParameterInput> target,
        voc_detection_dataset* ds,
        unsigned int bs){

    std::list<voc_detection_dataset::pattern> L;
    ds->get_batch(L, bs);
    
    unsigned int cnt=0;
    BOOST_FOREACH(voc_detection_dataset::pattern& pat, L) {
        input->data()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()]] = pat.img;
        ignore->data()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()]] = pat.ign;
        target->data()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()]] = pat.tch;
        cnt ++;
    }
}

int main(int argc, char **argv)
{
    // initialize cuv library   
    cuv::initialize_mersenne_twister_seeds();

    voc_detection_dataset ds("/home/local/datasets/VOC2011/voc_detection_trainval.txt", "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
   
    // number of simultaneously processed items
    unsigned int bs=16;

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input and target correctly
    // \c load_batch will put values in it.
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[bs][3][172*172],"input"));
    boost::shared_ptr<ParameterInput> ignore(
            new ParameterInput(cuv::extents[bs][20][172*172],"ignore"));
    boost::shared_ptr<ParameterInput> target(
            new ParameterInput(cuv::extents[bs][20][172*172],"target"));

    obj_detector od(5,16,5,20);
    od.init(input,ignore,target);

    std::vector<Op*> params = od.params();

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od.get_loss(),        "total loss");

    std::cout << std::endl << " Training phase: " << std::endl;
    {
        // create a \c gradient_descent object that derives the logistic loss
        // w.r.t. \c params and has learning rate 0.1f
        gradient_descent gd(od.get_loss(),0,params,0.0000001f / (43*43));
        
        // register the monitor so that it receives learning events
        gd.register_monitor(mon);

        // before each batch, load data into \c input
        gd.before_batch.connect(boost::bind(load_batch,input, ignore, target,&ds, bs));

        gd.after_batch.connect(
                boost::bind(&monitor::simple_logging, &mon));
        //gd.after_batch.connect(std::cout << ll::constant("\n"));
        
        // the number of batches is constant in our case (but has to be supplied as a function)
        gd.current_batch_num.connect(boost::bind(&voc_detection_dataset::trainset_size, &ds));
        
        // do mini-batch learning for at most 10 epochs, or 10 minutes
        // (whatever comes first)
        gd.minibatch_learning(10, 10*60); // 10 minutes maximum
    }
    std::cout <<  std::endl;

     
    return 0;
}





