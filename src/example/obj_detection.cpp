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

#define USE_BIOID 0

/**
 * load a batch from the dataset
 */
struct voc_loader{
    std::list<voc_detection_dataset::pattern> L;

    void done( boost::shared_ptr<Sink> output, voc_detection_dataset* ds){
        unsigned int cnt=0;
        BOOST_FOREACH(voc_detection_dataset::pattern& pat, L) {
            pat.result = output->cdata()[cuv::indices[cnt][cuv::index_range()][cuv::index_range()][cuv::index_range()]];
            cnt ++;
        }
        ds->save_results(L);
    }
    void load_batch(
            boost::shared_ptr<ParameterInput> input,
            boost::shared_ptr<ParameterInput> ignore,
            boost::shared_ptr<ParameterInput> target,
            voc_detection_dataset* ds,
            unsigned int bs){
        std::cout << "." << std::flush;

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

/**
 * load a batch from the dataset
 */
void load_batch_bioid(
        boost::shared_ptr<ParameterInput> input,
        boost::shared_ptr<ParameterInput> ignore,
        boost::shared_ptr<ParameterInput> target,
        cuv::tensor<float,cuv::host_memory_space>* data,
        cuv::tensor<float,cuv::host_memory_space>* labels,
        unsigned int bs, unsigned int batch){
    input->data() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    target->data() = (*labels)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    unsigned int imgX = sqrt(data->shape(1));
    unsigned int tgtX = sqrt(labels->shape(1)/2);
    input->data() .reshape(cuv::extents[bs][1][imgX][imgX]);
    target->data().reshape(cuv::extents[bs][2][tgtX][tgtX]);
    ignore->data() = 1.f;
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
    cuv::initCUDA(1);
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;
    std::string serialization_file = "obj_detector-%05d.ser";
    
    // number of simultaneously processed items
    unsigned int bs=32;

#if USE_BIOID
    bioid_dataset ds;
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[bs][1][120][120],"input"));
    boost::shared_ptr<ParameterInput> ignore(
            new ParameterInput(cuv::extents[bs][2][30][30],"ignore"));
    boost::shared_ptr<ParameterInput> target(
            new ParameterInput(cuv::extents[bs][2][30][30],"target"));
#else
    voc_detection_dataset ds("/home/local/datasets/VOC2011/voc_detection_train.txt", "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[bs][3][128][128],"input"));
    boost::shared_ptr<ParameterInput> ignore(
            new ParameterInput(cuv::extents[bs][20][128][128],"ignore"));
    boost::shared_ptr<ParameterInput> target(
            new ParameterInput(cuv::extents[bs][20][128][128],"target"));
#endif

    boost::shared_ptr<obj_detector> od;

    bool load_old_model_and_drop_to_python = argc > 1;
    if(! load_old_model_and_drop_to_python){
        // create new network
        od.reset( new obj_detector(7,32,7,64) );
        od->init(input,ignore,target);
    }else{
        //bs = 1;
        input.reset();
        ignore.reset();
        target.reset();
        od = deserialize_from_file<obj_detector>(serialization_file, boost::lexical_cast<int>(argv[1]));
        input  = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "input")->shared_from_this());
        target = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "target")->shared_from_this());
        ignore = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "ignore")->shared_from_this());
    }


    std::vector<Op*> params = od->params();
    if(load_old_model_and_drop_to_python){
        //load_batch(input, ignore, target, &ds, bs);
        initialize_python();
        export_ops();
        export_op("loss", od->get_loss());
#if USE_BIOID
        export_loadbatch(
                //boost::bind(load_batch_bioid,input, ignore, target,&ds.train_data, &ds.train_labels, bs, _1),
                //ds.train_data.shape(0)/ll::constant(bs));
                boost::bind(load_batch_bioid,input, ignore, target,&ds.test_data, &ds.test_labels, bs, _1),
                ds.test_data.shape(0)/ll::constant(bs));
#else
        voc_loader vl;
        export_loadbatch(
                boost::bind(&voc_loader::load_batch,&vl, input, ignore, target, &ds, bs),
                boost::bind(&voc_detection_dataset::trainset_size, &ds));
#endif
        embed_python();
        return 0;
    }

    // create a verbose monitor, so we can see progress 
    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_loss(),      "total loss");
    mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, od->get_f2(),   "F2");
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "tp", 1);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "tn", 2);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "fp", 3);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od->get_f2(),        "fn", 4);

    std::cout << std::endl << " Training phase: " << std::endl;
    {
        // create a \c gradient_descent object 
        rprop_gradient_descent gd(od->get_loss(),0,params);
        //momentum_gradient_descent gd(od->get_loss(),0,params, 0.1f);
        gd.decay_learnrate(.9999f);
        
        // register the monitor so that it receives learning events
        mon.register_gd(gd);

        // before each batch, load data into \c input
#if USE_BIOID
        gd.before_batch.connect(boost::bind(load_batch_bioid,input, ignore, target,&ds.train_data, &ds.train_labels, bs, _2));
#else
        voc_loader vl;
        gd.before_batch.connect(boost::bind(&voc_loader::load_batch, &vl,input, ignore, target,&ds, bs));
        gd.after_batch.connect(boost::bind(&voc_loader::done, &vl, od->m_output, &ds));
#endif

        gd.after_batch.connect(
                boost::bind(&monitor::simple_logging, &mon));
        gd.after_epoch.connect(std::cout << ll::constant("\n"));

#if !USE_BIOID
        gd.before_batch.connect(boost::bind(serialize_to_file<obj_detector>, serialization_file, od, _2, 100));
#endif
        gd.after_epoch.connect(boost::bind(serialize_to_file<obj_detector>, serialization_file, od, _1, 100));
        //gd.before_batch.connect(boost::bind(save_weights, od.get(), _2));
        
        // the number of batches 
#if USE_BIOID
        gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));
#else
        gd.current_batch_num = boost::bind(&voc_detection_dataset::trainset_size, &ds);
#endif
        //gd.current_batch_num.connect(ll::constant(128));
        
        // do mini-batch learning for at most 10 epochs, or until timeout
        // (whatever comes first)
        //gd.minibatch_learning(1000, 10000*60,1,false);
        gd.minibatch_learning(1, 10000*60,100,false);
        gd.minibatch_learning(1, 10000*60,200,false);
        gd.minibatch_learning(2, 10000*60,400,false);
        gd.minibatch_learning(3, 10000*60,800,false);
        gd.minibatch_learning(1000, 10000*60,0,false);
        //load_batch(input,ignore,target,&ds, bs);
        //gd.batch_learning(1000, 10000*60);
    }
    std::cout <<  std::endl;

     
    return 0;
}





