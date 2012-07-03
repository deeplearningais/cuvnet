// vim:ts=4:sw=4:et
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <tools/gradient_descent.hpp>

#include <tools/monitor.hpp>
#include <tools/matwrite.hpp>

#include <datasets/voc_detection.hpp>
#include <cuvnet/models/object_detector.hpp>
#include <cuvnet/op_io.hpp>

#include <tools/serialization_helper.hpp>
#include <tools/python_helper.hpp>

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

static unsigned int cnt=0;
void save_weights(obj_detector* ds, unsigned int epoch){
    if(cnt++ % 100 == 0){
        tofile((boost::format("conv1-%05d.npy") % cnt ).str(), ds->m_conv1_weights->data());
        tofile((boost::format("conv2-%05d.npy") % cnt ).str(), ds->m_conv2_weights->data());
    }
}

int main(int argc, char **argv)
{
    // initialize cuv library   
    cuv::initialize_mersenne_twister_seeds();
    std::string serialization_file = "obj_detector-%05d.ser";

    voc_detection_dataset ds("/home/local/datasets/VOC2011/voc_detection_train.txt", "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
   
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

    boost::shared_ptr<obj_detector> od;

    bool load_old_model_and_drop_to_python = argc > 1;
    if(! load_old_model_and_drop_to_python){
        // create new network
        od.reset( new obj_detector(7,32,7,32) );
        od->init(input,ignore,target);
    }else{
        od = deserialize_from_file<obj_detector>(serialization_file, boost::lexical_cast<int>(argv[1]));
    }


    std::vector<Op*> params = od->params();
    if(load_old_model_and_drop_to_python){
        load_batch(input, ignore, target, &ds, bs);
        initialize_python();
        export_ops();
        export_op("loss", od->get_loss());
        embed_python();
        return 0;
    }

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_loss(),        "total loss");

    std::cout << std::endl << " Training phase: " << std::endl;
    {
        // create a \c gradient_descent object 
        gradient_descent gd(od->get_loss(),0,params,0.0000001f, .000010f);
        
        // register the monitor so that it receives learning events
        gd.register_monitor(mon);

        // before each batch, load data into \c input
        gd.before_batch.connect(boost::bind(load_batch,input, ignore, target,&ds, bs));

        gd.after_batch.connect(
                boost::bind(&monitor::simple_logging, &mon));
        gd.after_epoch.connect(std::cout << ll::constant("\n"));

        gd.before_batch.connect(boost::bind(serialize_to_file<obj_detector>, serialization_file, od, _2, 100));
        gd.before_batch.connect(boost::bind(save_weights, od.get(), _2));
        
        // the number of batches 
        gd.current_batch_num.connect(boost::bind(&voc_detection_dataset::trainset_size, &ds));
        
        // do mini-batch learning for at most 10 epochs, or until timeout
        // (whatever comes first)
        gd.minibatch_learning(10, 10000*60,1,false); // 1h
    }
    std::cout <<  std::endl;

     
    return 0;
}





