// vim:ts=4:sw=4:et
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <tools/gradient_descent.hpp>

#include <tools/monitor.hpp>
#include <tools/matwrite.hpp>

#include <datasets/voc_detection.hpp>
#include <datasets/bioid.hpp>
#include <cuvnet/models/object_detector.hpp>
#include <cuvnet/op_io.hpp>

#include <tools/serialization_helper.hpp>
#include <tools/python_helper.hpp>
#include <tools/logging.hpp>

using namespace boost::assign;
using namespace cuvnet;
namespace ll = boost::lambda;

/**
 * load a batch from the dataset
 */
struct voc_loader{
    std::list<voc_detection_dataset::pattern> L;

    void done( boost::shared_ptr<Sink> output, voc_detection_dataset* ds){
        unsigned int cnt=0;
        BOOST_FOREACH(voc_detection_dataset::pattern& pat, L) {
            //pat.result = output->cdata()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()][cuv::index_range()]];
            cnt ++;
        }
        //ds->save_results(L);
    }
    void load_batch(
            boost::shared_ptr<ParameterInput> input,
            boost::shared_ptr<ParameterInput> ignore,
            boost::shared_ptr<ParameterInput> target,
            voc_detection_dataset* ds,
            unsigned int bs){
        L.clear();
        ds->get_batch(L, bs);

        unsigned int cnt=0;
        BOOST_FOREACH(voc_detection_dataset::pattern& pat, L) {
            input ->data()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()][cuv::index_range()]] = pat.img;
            ignore->data()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()][cuv::index_range()]] = pat.ign;
            target->data()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()][cuv::index_range()]] = pat.tch;
            cnt ++;
        }
    }
};

int main(int argc, char **argv)
{
    // initialize cuv library   
    cuv::initCUDA(0);
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;
    std::string serialization_file = "obj_detector-%05d.ser";
    
    // number of simultaneously processed items
    unsigned int bs=32;

    //voc_detection_dataset ds("/home/local/datasets/VOC2011/voc_detection_train_small.txt", "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
    voc_detection_dataset ds("/home/local/datasets/VOC2011/voc_detection_train.txt", "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[bs][3][128][128],"input"));
    boost::shared_ptr<ParameterInput> ignore;
    boost::shared_ptr<ParameterInput> target;

    boost::shared_ptr<obj_detector> od;

    bool load_old_model_and_drop_to_python = argc > 1;
    if(! load_old_model_and_drop_to_python){
        // create new network
        od.reset( new obj_detector(5,32,7,64) );
        od->init(input, 1);
    }else{
        //bs = 1;
        input.reset();
        ignore.reset();
        target.reset();
        od = deserialize_from_file<obj_detector>(serialization_file, boost::lexical_cast<int>(argv[1]));
        input  = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "input")->shared_from_this());
    }
    target = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "target")->shared_from_this());
    ignore = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "ignore")->shared_from_this());

    valid_shape_info vsi(input, od->hl3);
    ds.set_output_properties(vsi.scale_h, vsi.scale_w, vsi.crop_h, vsi.crop_w);


    std::vector<Op*> params = od->params();
    if(load_old_model_and_drop_to_python){
        initialize_python();
        export_ops();
        export_op("loss", od->get_loss());
        voc_loader vl;
        vl.load_batch(input, ignore, target, &ds, bs);
        export_loadbatch(
                boost::bind(&voc_loader::load_batch,&vl, input, ignore, target, &ds, bs),
                boost::bind(&voc_detection_dataset::trainset_size, &ds));
        embed_python();
        return 0;
    }

    // create a verbose monitor, so we can see progress 
    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_loss(),      "total loss");
    mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, od->get_f2(),   "F2", 0);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "tp", 1);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "tn", 2);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "fp", 3);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "fn", 4);

    std::cout << std::endl << " Training phase: " << std::endl;
    {
        // create a \c gradient_descent object 
        //rprop_gradient_descent gd(od->get_loss(),0,params);
        momentum_gradient_descent gd(od->get_loss(),0,params, 0.01f);
        convergence_checker cs(gd, boost::bind(&monitor::mean, &mon, "total loss"));
        cs.decrease_lr(10000, .95f);
        
        // register the monitor so that it receives learning events
        mon.register_gd(gd);

        // before each batch, load data into \c input
        voc_loader vl;
        gd.before_batch.connect(boost::bind(&voc_loader::load_batch, &vl,input, ignore, target,&ds, bs));
        gd.after_batch.connect(boost::bind(&voc_loader::done, &vl, od->m_output, &ds));

        gd.after_batch.connect(
                boost::bind(&monitor::simple_logging, &mon));
        gd.after_epoch.connect(std::cout << ll::constant("\n"));

        serialize_to_file<obj_detector>(serialization_file, od, 0, 1);
        gd.after_weight_update.connect(boost::bind(serialize_to_file<obj_detector>, serialization_file, od, _1, 640));
        gd.after_weight_update.connect(boost::bind(&obj_detector::project_to_allowed_region, &*od));
        
        // the number of batches 
        //gd.current_batch_num = boost::bind(&voc_detection_dataset::trainset_size, &ds);
        gd.current_batch_num = ll::constant(85);
        //gd.current_batch_num.connect(ll::constant(128));
        
        gd.minibatch_learning(1E6, 10000*60,1,false);
        //gd.minibatch_learning(1, 10000*60,8,false);
        //gd.minibatch_learning(2, 10000*60,16,false);
        //gd.minibatch_learning(4, 10000*60,32,false);
        //gd.minibatch_learning(1E6, 10000*60,64,false);
        //load_batch(input,ignore,target,&ds, bs);
        //gd.batch_learning(1000, 10000*60);
    }
    std::cout <<  std::endl;

     
    return 0;
}





