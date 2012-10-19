// vim:ts=4:sw=4:et
#include <sys/syscall.h> /* for pid_t, syscall, SYS_gettid */
#include <boost/bind.hpp>
#include <boost/thread.hpp>
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
#include <tools/network_communication.hpp>

using namespace boost::assign;
using namespace cuvnet;
namespace ll = boost::lambda;

std::string g_serialization_file("obj_detector-%06d.ser");

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

void async_client(unsigned int dev, int total_devs, bool checker, const std::string& key){
    cuv::initCUDA(dev);
    cuv::initialize_mersenne_twister_seeds(42);
    srand48(42);
    srand(42);


    int bs = 32;
    boost::shared_ptr<ParameterInput> ignore, target, input( new ParameterInput(cuv::extents[bs][3][176][176],"input"));
    boost::shared_ptr<obj_detector> p_od( new obj_detector(5,32,7,64));
    obj_detector& od = *p_od;
    od.init(input, 1);
    target = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od.get_loss(), "target")->shared_from_this());
    ignore = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od.get_loss(), "ignore")->shared_from_this());


    srand48(time(NULL) + (pid_t)syscall(SYS_gettid));
    srand(time(NULL) + (pid_t)syscall(SYS_gettid));

    valid_shape_info vsi(input, od.hl3);

    //voc_detection_dataset ds("/home/local/datasets/VOC2011/voc_detection_train_small.txt", "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
    voc_detection_dataset ds(
            "/home/local/datasets/VOC2011/voc_detection_train.txt", 
            "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
    ds.set_output_properties(vsi.scale_h, vsi.scale_w, vsi.crop_h, vsi.crop_w);
    ds.switch_dataset(voc_detection_dataset::SS_TRAIN);

    std::vector<Op*> params = od.params();

    monitor mon(checker);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, od.get_loss(),      "total loss");

    if(checker){
        mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, od.get_f2(),   "F2", 0);
        mon.add(monitor::WP_SCALAR_EPOCH_STATS, od.get_f2(),        "tp", 1);
        mon.add(monitor::WP_SCALAR_EPOCH_STATS, od.get_f2(),        "tn", 2);
        mon.add(monitor::WP_SCALAR_EPOCH_STATS, od.get_f2(),        "fp", 3);
        mon.add(monitor::WP_SCALAR_EPOCH_STATS, od.get_f2(),        "fn", 4);
    }



    {
        //typedef momentum_gradient_descent gd_base_t;
        typedef gradient_descent gd_base_t;
        typedef cuvnet::diff_recording_gradient_descent<gd_base_t>  gd_t;

        //rprop_gradient_descent gd(od.get_loss(),0,params);
        gd_t gd(od.get_loss(),0,params, 0.001f);
        //gd_t gd(od.get_loss(),0,params, 1.f);

        convergence_checker cs(gd, boost::bind(&monitor::mean, &mon, "total loss"), .95f, 400);
        cs.decrease_lr(10000, .95f);
        
        mon.register_gd(gd);
        cuvnet::network_communication::client clt( 
                "131.220.7.92","objdet","async",
                "client-"+boost::lexical_cast<std::string>(dev));

        int sync_freq = 18;
        int f2 = sync_freq / total_devs;
        const std::vector<Op*> params = od.params();
        cuvnet::network_communication::param_synchronizer psync("sfinetune", clt, 
                    sync_freq, f2, 0,  0, params);
        if(total_devs > 1)
            gd.set_sync_function(boost::ref(psync));
        gd.after_epoch.connect(boost::bind(&cuvnet::network_communication::param_synchronizer::test_stop, &psync));
        gd.done_learning.connect(boost::bind(&cuvnet::network_communication::param_synchronizer::stop_coworkers, &psync));

        voc_loader vl;
        gd.before_batch.connect(boost::bind(&voc_loader::load_batch, &vl,input, ignore, target,&ds, bs));
        if(checker) {
            gd.after_batch.connect(boost::bind(&monitor::simple_logging, &mon));
            gd.after_epoch.connect(boost::bind(serialize_to_file<obj_detector>, g_serialization_file, p_od, _1, 20));
        }
        //gd.after_batch.connect(boost::bind(&voc_loader::done, &vl, od.m_output, &ds));
        //gd.after_weight_update.connect(boost::bind(&obj_detector::project_to_allowed_region, &*od));
        
        // the number of batches 
        gd.current_batch_num = ll::constant(119);
        
        gd.minibatch_learning(1E6, 10000*60,1,false);
    }

}

int main(int argc, char **argv)
{
    bool load_old_model_and_drop_to_python = argc > 1;
    if(load_old_model_and_drop_to_python){
        cuv::initCUDA(boost::lexical_cast<int>(argv[2]));
        // initialize cuv library   
        cuvnet::Logger log("load-session.xml");
        voc_detection_dataset ds("/home/local/datasets/VOC2011/voc_detection_train.txt", "/home/local/datasets/VOC2011/voc_detection_val.txt", true);
        boost::shared_ptr<ParameterInput> ignore, target, input;
        boost::shared_ptr<obj_detector> od = deserialize_from_file<obj_detector>(g_serialization_file, boost::lexical_cast<int>(argv[1]));
        input  = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "input")->shared_from_this());
        target = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "target")->shared_from_this());
        ignore = boost::dynamic_pointer_cast<ParameterInput>(get_node<ParameterInput>(od->get_loss(), "ignore")->shared_from_this());

        valid_shape_info vsi(input, od->hl3);
        ds.set_output_properties(vsi.scale_h, vsi.scale_w, vsi.crop_h, vsi.crop_w);
        ds.switch_dataset(voc_detection_dataset::SS_TRAIN);

        // number of simultaneously processed items
        unsigned int bs=32;
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



    std::vector<unsigned int> devs;
    //devs.push_back(0);
    devs.push_back(1);
    devs.push_back(2);
    devs.push_back(3);

    // start server
    std::string db = "objdet";
    std::string key = devs.size() == 1 ? "sync" : "async";
    cuvnet::Logger __log("obj_det-"+key+".xml");
    log4cxx::LoggerPtr log = log4cxx::Logger::getLogger("main");

    //cuvnet::network_communication::momentum_merger mrg(0.9);
    cuvnet::network_communication::adagrad_merger mrg;
    cuvnet::network_communication::server s("131.220.7.92", db, key, &mrg);
    if(devs.size() > 1) {
        LOG4CXX_WARN(log, "starting up server for key " << key);
        s.cleanup();
        //boost::thread* server_thread = new boost::thread(boost::bind(&cuvnet::network_communication::server::run, &s, 1, -1));
        new boost::thread(boost::bind(&cuvnet::network_communication::server::run, &s, 1, -1));
    }else{
        LOG4CXX_WARN(log, "no server for key " << key);
    }

    std::vector<boost::shared_ptr<boost::thread> >  threads(devs.size());

    for (unsigned int i = 0; i < devs.size(); ++i){
        threads[i] = boost::make_shared<boost::thread>(boost::bind(async_client, devs[i], devs.size(), i==0, key));
    }


    for (unsigned int i = 0; i < devs.size(); ++i)
        threads[i]->join();

    std::cout <<  std::endl;
    return 0;
}





