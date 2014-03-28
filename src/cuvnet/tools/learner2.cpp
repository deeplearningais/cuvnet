#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/min.hpp>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/filesystem.hpp>

#include <boost/algorithm/string.hpp> 

#include <cuvnet/tools/serialization_helper.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/tools/exception_tracer.hpp>
#include <cuvnet/tools/network_communication.hpp>
#include <log4cxx/ndc.h>

#include <cuv/basics/io.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <cuvnet/common.hpp>
#include "learner2.hpp"

namespace bfs = boost::filesystem;

typedef boost::property_tree::ptree ptree;


BOOST_SERIALIZATION_ASSUME_ABSTRACT(cuvnet::model)

namespace 
{
    log4cxx::LoggerPtr g_log_learner2(log4cxx::Logger::getLogger("learner2"));
    log4cxx::LoggerPtr g_log_xval2(log4cxx::Logger::getLogger("xval2"));
    log4cxx::LoggerPtr g_log_mslearner(log4cxx::Logger::getLogger("multistage_learner"));
    cuvnet::cv_mode g_cm[] = {cuvnet::CM_TRAIN, cuvnet::CM_VALID, cuvnet::CM_TEST, cuvnet::CM_TRAINALL};
}

/* For streaming, it is essential to turn off object tracking, since multiple
 * objects may be saved under the same address at different times.
 * 
 * Boost does not currently interpret the no_tracking parameter to the archive
 * constructor. The mechanism via 
 *   BOOST_CLASS_TRACKING(my_class, boost::serialization::track_never)
 * is broken for shared_ptr (you get static asserts).
 * The method from here:
 *   http://stackoverflow.com/questions/15614093/derived-class-serialization-without-class-tracking-in-boost-c/16018170#16018170
 * approaches the problem by injecting no_tracking support into
 * an existing archive class:
 */
namespace boost {
    namespace archive {
        namespace detail {
            template<>
                bool oserializer<binary_oarchive, cuvnet::matrix >::tracking(const unsigned int f /* flags */) const {
                    return !(f & no_tracking);
                }
            template<>
                bool oserializer<binary_oarchive, cuvnet::host_matrix >::tracking(const unsigned int f /* flags */) const {
                    return !(f & no_tracking);
                }
        }}}

namespace cuvnet
{
    namespace schedules
    {
        /*****************************************
         * linear_learnrate_schedule
         *****************************************/
        linear_learnrate_schedule::linear_learnrate_schedule(gradient_descent* _gd, float begin, float end, int epochs)
            :initial(begin), final(end), duration(epochs), gd(_gd)
        {
            con = gd->before_epoch.connect(boost::ref(*this));
        }
        void linear_learnrate_schedule::operator()(unsigned int epoch, unsigned int wups)
        {
            gd->set_learnrate(
                    std::max(final, initial + 
                        (final - initial) * epoch  / (float)duration));
        }
        /*****************************************
         * exponential_learnrate_schedule
         *****************************************/
        exponential_learnrate_schedule::exponential_learnrate_schedule(gradient_descent* _gd, float begin, float end, int epochs, float t0_)
            :initial(begin), final(end), duration(epochs), t0(t0_), gd(_gd)
        {
            alpha = std::log(final/initial) / std::log(t0/(t0 + duration));
            eta0  = initial * std::pow(t0, alpha);
            LOG4CXX_WARN(g_log_learner2, "Setting up exponential learnrate schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration
                    << ", alpha:"<<alpha
                    << ", eta0:"<<eta0
                    << ", t0:"<<t0<<")");
            con = gd->before_epoch.connect(boost::ref(*this));
        }
        void exponential_learnrate_schedule::operator()(unsigned int epoch, unsigned int wups)
        {
            gd->set_learnrate(
                    std::max(final, 
                        eta0 / std::pow(t0 + epoch, alpha)
                        ));
        }
        /*****************************************
         * linear_momentum_schedule
         *****************************************/
        linear_momentum_schedule::linear_momentum_schedule(momentum_gradient_descent* _gd, float begin, float end, int epochs)
            :initial(begin), final(end), duration(epochs),gd(_gd)
        {
            con = gd->before_epoch.connect(boost::ref(*this));
        }
        void linear_momentum_schedule::operator()(unsigned int epoch, unsigned int wups)
        {
            gd->set_momentum(
                    std::min(final, initial + 
                        (final - initial) * epoch  / (float)duration));
        }
    };

    /*****************************************
     * record_optimal_training_loss 
     *****************************************/
    record_optimal_training_loss::record_optimal_training_loss(early_stopper& es, monitor& _mon)
            :current_training_loss(1e9), best_training_loss(1e9), mon(&_mon){
            con0 = es.before_early_stopping_epoch.connect(boost::bind(&record_optimal_training_loss::before_early_stopping_epoch, this, _1));
            con1 = es.improved.connect(boost::bind(&record_optimal_training_loss::improved, this));
        }
    void record_optimal_training_loss::before_early_stopping_epoch(unsigned int){
        current_training_loss = mon->mean("loss");
        if(current_training_loss != current_training_loss)
            current_training_loss = INT_MAX;
    }
    void record_optimal_training_loss::improved(){
        best_training_loss = current_training_loss;
    }

    stop_when_target_loss_reached::stop_when_target_loss_reached(gradient_descent& gd, monitor& _mon, float tloss)
        :target_loss(tloss), mon(&_mon){
            con = gd.after_epoch.connect(boost::ref(*this));
        }
    void stop_when_target_loss_reached::operator()(unsigned int current_epoch, unsigned int wups){
        float f = mon->mean("loss");
        if(f != f)
            throw gradient_descent_stop();
        if(f <= target_loss)
            throw gradient_descent_stop();
    }

#define DRGD diff_recording_gradient_descent
#define WRGD wup_recording_gradient_descent
#define MAKE_GD(BASETYPE, ...) \
    boost::shared_ptr<gradient_descent> gd; \
    if(drec)  \
      gd = boost::make_shared<DRGD<BASETYPE> >(__VA_ARGS__);\
    else if(wrec) \
      gd = boost::make_shared<WRGD<BASETYPE> >(__VA_ARGS__);\
    else  \
      gd = boost::make_shared<BASETYPE>(__VA_ARGS__);\

    /*****************************************
     * Learner2
     *****************************************/
    boost::shared_ptr<gradient_descent> 
    learner2::get_gradient_descent(model& m, const ptree& cfg){
        std::string typ = cfg.get("type", "plain");
        bool drec = cfg.get("diff_rec", false);
        bool wrec = cfg.get("wup_rec.active", false);
        if(drec){
            LOG4CXX_WARN(g_log_learner2, "Setting up delta recording for gradient_descent");
        }
        if(wrec){
            LOG4CXX_WARN(g_log_learner2, "Setting up wup recording for gradient_descent");
        }
        float initial_learnrate = cfg.get("learnrate", 0.01);
        float initial_momentum = cfg.get("momentum", 0.9);
        float l2decay = cfg.get("l2decay", 0.0);
        int verbosity = cfg.get("verbosity", 0);
        unsigned int start_epoch = cfg.get("start_epoch", 0);
        if(typ == "plain"){
            LOG4CXX_WARN(g_log_learner2, "Creating Plain GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            MAKE_GD(gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(typ == "rmsprop"){
            float delta = cfg.get("delta", 0.01f);
            float grad_avg = cfg.get("grad_avg", 0.9f);
            float l1decay = cfg.get("l1decay", 0.f);
            LOG4CXX_WARN(g_log_learner2, "Creating RMSProp GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << "grad_avg:" << grad_avg << ")");
            MAKE_GD(rmsprop_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay, delta, grad_avg, l1decay);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(typ == "rprop"){
            LOG4CXX_WARN(g_log_learner2, "Creating RPROP GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            MAKE_GD(rprop_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay);
            gd->set_epoch(start_epoch);
            gd->set_update_every(0);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(typ == "momentum"){
            LOG4CXX_WARN(g_log_learner2, "Creating Momentum GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", momentum: "<< initial_momentum << ")");
            MAKE_GD(momentum_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay, initial_momentum);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(typ == "adagrad"){
            float winsize = cfg.get("winsize", INT_MAX);
            float l1_penalty = cfg.get("l1_penalty", 0.f);
            float delta = cfg.get("delta", 0.01f);
            LOG4CXX_WARN(g_log_learner2, "Creating AdaGrad GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", delta:"<< delta 
                << ", winsize:" << winsize << ", l1_penalty:" << l1_penalty << ")");
            MAKE_GD(adagrad_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay,  delta, winsize, l1_penalty);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(typ == "na_rmsprop"){
            LOG4CXX_WARN(g_log_learner2, "Creating OriginalNAG GD (initial_learnrate:"<<initial_learnrate<<", momentum:"<< initial_momentum  << ")");
            MAKE_GD(na_rmsprop_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay,  initial_momentum, 0.9f);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else{
            throw std::runtime_error("unknown/not implemented gd_type: " + typ);
        }
    }

    boost::shared_ptr<early_stopper>
    learner2::get_early_stopper(gradient_descent& gd, monitor& mon, const ptree& cfg){
        boost::shared_ptr<early_stopper> es;
        bool active = cfg.get("active", true);
        if(!active)
            return es;
        std::string watch = cfg.get("watch", "cerr");
        float thresh = cfg.get("thresh", 0.995);
        int every = cfg.get("every", 1);
        float multiply = cfg.get("multiply", 2.f);
        int boxfilter = cfg.get("boxfilter", 1);
        int patience = cfg.get("patience", 100);
        LOG4CXX_WARN(g_log_learner2, "Setting up Early Stopper (watch:"<<watch
                <<", thresh:"<< thresh
                << ", every: "<< every 
                << ", multiply:"<<multiply
                << ", boxfilter:" << boxfilter 
                << ", patience:" << patience<< ")");
        es.reset(new early_stopper(gd, boost::bind(&monitor::mean, &mon, watch), thresh, every, multiply, boxfilter));
        es->before_early_stopping_epoch.connect(boost::bind(&learner2::_switch_dataset, this, CM_VALID, 0));
        es->after_early_stopping_epoch.connect(boost::bind(&learner2::_switch_dataset, this, CM_TRAIN, 0));

        es->before_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_VALID, 0));
        es->after_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_TRAIN, 0));
        es->set_patience(patience);
        return es;
    }

    boost::shared_ptr<convergence_checker>
    learner2::get_convergence_checker(gradient_descent& gd, monitor& mon, const ptree& cfg){
        boost::shared_ptr<convergence_checker> cc;
        bool active = cfg.get("active", true);
        if (!active)
            return cc;
        std::string watch = cfg.get("watch", "cerr");
        float thresh = cfg.get("thresh", 0.995);
        int min_wups = cfg.get("min_wups", 10);
        float patience_inc_fact = cfg.get("patience_inc_fact", 1.2);
        LOG4CXX_WARN(g_log_learner2, "Setting up Convergence Checker ("
                <<  "watch: " << watch
                << " thresh: " << thresh
                << " min_wups: " << min_wups
                << " patience_inc_fact: " << patience_inc_fact
                );
        cc.reset(new convergence_checker(gd, boost::bind(&monitor::mean, &mon, watch), thresh, min_wups, patience_inc_fact));
        return cc;
    }

    boost::shared_ptr<monitor> 
    learner2::get_monitor(model& m, const ptree& cfg){
        bool verbose = cfg.get("verbose", true);
        boost::shared_ptr<monitor> mon = boost::make_shared<monitor>(verbose);
        mon->add(monitor::WP_SCALAR_EPOCH_STATS, m.loss(), "loss");
        if(m.error())
            mon->add(monitor::WP_SCALAR_EPOCH_STATS, m.error(), "cerr");
        m.register_watches(*mon);
        return mon;
    }
    
    void
    learner2::register_validation_batchsize(model& m , early_stopper& es, const ptree& cfg){
        unsigned int bs_train = cfg.get("batchsize", -1);
        unsigned int bs_valid = cfg.get("batchsize_valid", bs_train);

        if(bs_train != bs_valid){
            es.before_early_stopping_epoch.connect(boost::bind(&model::set_batchsize, &m, bs_valid));
            es.before_early_stopping_epoch.connect(boost::bind(&gradient_descent::repair_swiper, m_gd));
            es.after_early_stopping_epoch.connect(boost::bind(&model::set_batchsize, &m, bs_train));
            es.after_early_stopping_epoch.connect(boost::bind(&gradient_descent::repair_swiper, m_gd));
        }
    }

    void 
    learner2::load_batch(model* m, unsigned int epoch, unsigned int batch){
    }

    void 
    learner2::_load_batch(model* m, unsigned int epoch, unsigned int batch){
        load_batch(m, epoch, batch);
    }

    void 
    learner2::before_predict(model* m, gradient_descent&, const ptree&){
    }

    void 
    learner2::before_learning(model* m, gradient_descent&, cuvnet::early_stopper* es, const ptree&){
    }

    unsigned int 
    learner2::_n_batches(unsigned int batchsize){
        return n_batches(batchsize);
    }

    unsigned int 
    learner2::n_batches(unsigned int batchsize){
        return 1;
    }

    void learner2::_switch_dataset(cv_mode mode, int split){
        switch_dataset(mode, split);
    }

    void learner2::switch_dataset(cv_mode mode, int split){
    }

    learner2::~learner2(){};

    boost::shared_ptr<schedules::hyperparam_schedule> 
    learner2::get_learnrate_schedule(gradient_descent& gd, int max_epochs, ptree cfg){
        boost::shared_ptr<schedules::hyperparam_schedule> hs;
        std::string schedule = cfg.get("type", "constant");
        if(schedule == "constant")
            return hs;
        if(schedule == "linear"){
            float initial = cfg.get("initial", gd.learnrate());
            float final   = cfg.get("final", initial * 0.01f);
            int   duration = cfg.get("duration", max_epochs);
            LOG4CXX_WARN(g_log_learner2, "Setting up linear learnrate schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration<<")");
            hs.reset(new schedules::linear_learnrate_schedule(&gd, initial, final, duration));
            return hs;
        }
        if(schedule == "exponential"){
            float initial = cfg.get("initial", gd.learnrate());
            float final   = cfg.get("final", 0.000001f);
            float t0   = cfg.get("t0", 10.f);
            int   duration = cfg.get("duration", max_epochs);
            hs.reset(new schedules::exponential_learnrate_schedule(&gd, initial, final, duration, t0));
            return hs;
        }
        throw std::runtime_error("unknown learnrate schedule: " + schedule);
    }

    boost::shared_ptr<schedules::hyperparam_schedule> 
    learner2::get_momentum_schedule(gradient_descent& pgd, int max_epochs, ptree cfg){
        boost::shared_ptr<schedules::hyperparam_schedule> hs;

        momentum_gradient_descent* gd = dynamic_cast<momentum_gradient_descent*>(&pgd);
        if(!gd)
            return hs;

        std::string schedule = cfg.get("type", "constant");
        if(schedule == "constant")
            return hs;
        if(schedule == "linear"){
            float initial = cfg.get("initial", gd->momentum());
            float final   = cfg.get("final", 0.9f);
            int   duration = cfg.get("duration", max_epochs);
            LOG4CXX_WARN(g_log_learner2, "Setting up linear momentum schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration<<")");
            hs.reset(new schedules::linear_momentum_schedule(gd, initial, final, duration));
            return hs;
        }
        throw std::runtime_error("unknown learnrate schedule: " + schedule);
    }

    ptree 
    learner2::continue_learning_until_previous_loss_reached(model& m, const ptree& cfg, const ptree& result){
        // - stop when target loss reached
        // - set max_epochs=old_epoch PLUS new max_epochs
        // - let gradient_descent START from a certain number of epochs
        ptree cfg2 = cfg;
        cfg2.put("early_stopper.active", false);
        cfg2.put("min_loss_stopper.active", true);
        cfg2.put("min_loss_stopper.target_loss", result.get<float>("gd.early_stopper.optimal_training_error"));
        cfg2.put("gd.start_epoch",    result.get<unsigned int>("gd.result_epoch"));
        cfg2.put("gd.max_epochs", 8 * result.get<unsigned int>("gd.result_epoch"));
        ptree res = fit(m, cfg2);
        return res;
    }
    ptree 
    learner2::learn_until_previous_loss_reached(model& m, const ptree& cfg, const ptree& result){
        float target_loss = result.get<float>("gd.early_stopper.optimal_training_error");
        LOG4CXX_WARN(g_log_learner2, "multistage_learner: learn_until_previous_loss_reached, target_loss:"<<target_loss);
        ptree cfg2 = cfg;
        cfg2.put("early_stopper.active", false);
        cfg2.put("min_loss_stopper.active", true);
        cfg2.put("min_loss_stopper.target_loss", target_loss);
        ptree res = fit(m, cfg2);
        return res;
    }

    unsigned int learner2::n_splits()const{
        return 1;
    }
    void 
        learner2::save_model(boost::shared_ptr<model>& m, std::string filename){
            namespace bar= boost::archive;
            std::ofstream f(filename.c_str());
            bar::binary_oarchive oa(f);
            register_objects(oa);
            oa << m;
        }
    void 
        learner2::load_model(boost::shared_ptr<model>& m, std::string filename){
            namespace bar= boost::archive;
            std::ifstream f(filename.c_str());
            bar::binary_iarchive ia(f);
            register_objects(ia);
            ia >> m;
        }

    ptree 
    crossvalidator2::fit(learner2& lrn, boost::shared_ptr<model> mptr, const ptree& cfg){
        model& m = *mptr;
        unsigned int n_splits = lrn.n_splits();
        ptree result;
        using namespace boost::accumulators;
        namespace ba = boost::accumulators;
        accumulator_set<double, 
            features<tag::mean, tag::median, tag::variance, tag::min> > s_valperf;
        accumulator_set<double, 
            features<tag::mean, tag::median, tag::variance, tag::min> > s_testperf;
        unsigned int best_split = 0;
        bfs::path bestfile;
        ptree best_result;
        std::string measure_path = cfg.get("xval.measure_path", "gd.early_stopper.best_perf");
        for(unsigned int split = 0; split < n_splits; split ++){
            log4cxx::NDC ndc("split_"+boost::lexical_cast<std::string>(split));
            m.reset_params();
            lrn._switch_dataset(CM_TRAIN, split);
            ptree res = lrn.fit(m, cfg);
            if(split == 0 || ba::min(s_valperf) > res.get<float>(measure_path)){
                bestfile =  bfs::path(res.get<std::string>("path")) / "after_train.ser";
                lrn.save_model(mptr, bestfile.string());
                best_split = split;
                best_result = res;
            }

            // evaluate on test set, (not used for model selection!)
            lrn._switch_dataset(CM_TEST);
            ptree pres = lrn.predict(m, cfg);
            res.add_child("testset", pres);


            // record fold result
            result.add_child("xval.folds.fold_" + boost::lexical_cast<std::string>(split), res);
            s_valperf(res.get<float>(measure_path)); 

            boost::optional<float> predict_loss 
                = pres.get_optional<float>("cerr_mean");
            if(predict_loss)
                s_testperf(*predict_loss); 
            else
                s_testperf(pres.get<float>("loss_mean")); 
            // TODO stop evaluating if too bad in comparison with current best?
        }
        result.put("xval.val_mean", ba::mean(s_valperf));
        result.put("xval.val_var", ba::variance(s_valperf));
        result.put("xval.test_mean", ba::mean(s_testperf));
        result.put("xval.test_var", ba::variance(s_testperf));
        result.put_child("xval.best_fold", best_result);

        // retrain on all, training + validation, if required.
        if(cfg.get("xval.retrain_all", false)){
            log4cxx::NDC ndc("retrain_all");
            float current_loss = ba::mean(s_valperf);
            float prev_loss = cfg.get("xval.retrain_all_thresh", INT_MAX);
            if(current_loss < prev_loss){
                LOG4CXX_WARN(g_log_xval2, "retrain on TRAINVAL, current_loss:"<<current_loss<<", prev_loss:"<<prev_loss);
                m.reset_params();
                ptree cfg2 = cfg;
                cfg2.put("train_dataset", (int)CM_TRAINALL); // required for multi-stage learning
                ptree tres = lrn.learn_until_previous_loss_reached(m, cfg2, best_result);

                lrn._switch_dataset(CM_TEST);
                ptree pres = lrn.predict(m, cfg2);
                result.put_child("xval.retrain_all.training", tres);
                result.put_child("xval.retrain_all.test", pres);
            }else{
                LOG4CXX_WARN(g_log_xval2, "NO retrain on TRAINVAL, current_loss:"<<current_loss<<", prev_loss:"<<prev_loss);
            }
        }

        return result;
    }

    ptree learner2::predict(model& m, const ptree& cfg){
        m_mon
            = get_monitor(m, cfg.get_child("monitor", ptree()));

        int batch_size = cfg.get("batchsize", -1);
        gradient_descent gd(m.loss(), 0, std::vector<Op*>(), 0.0, 0.0);
        this->before_predict(&m, gd, cfg);
        m_mon->register_gd(gd);
        if(batch_size < 0){
            _load_batch(&m, 0, 0);
            gd.batch_learning(1, INT_MAX);
        }else {
            int batch_size_test = cfg.get("batchsize_test", batch_size);
           
            if(batch_size_test != batch_size){
                m.set_batchsize(batch_size_test);
                gd.repair_swiper();
            }

            gd.before_batch.connect(boost::bind(&learner2::_load_batch, this, &m, _1, _2));
            gd.current_batch_num = boost::bind(&learner2::_n_batches, this, batch_size);
            gd.minibatch_learning(1, INT_MAX, false); // don't shuffle
        }
        ptree result;
        if(m_mon->has("cerr")){
            result.put("cerr_mean", m_mon->mean("cerr"));
            result.put("cerr_var", m_mon->var("cerr"));
        }else{
            result.put("loss_mean", m_mon->mean("loss"));
            result.put("loss_var", m_mon->var("loss"));
        }
        return result;
    }

    ptree 
    learner2::fit(model& m, const ptree& cfg)
    {
        std::string basepath = cfg.get("basepath", ".");
        bfs::path tmppath = bfs::path(basepath) / cfg.get("path", ".");
        boost::filesystem::create_directories(tmppath);
        std::string uuid = boost::lexical_cast<std::string>(boost::uuids::uuid(boost::uuids::random_generator()()));

        m_gd 
            = get_gradient_descent(m, cfg.get_child("gd"));

        m_mon
            = get_monitor(m, cfg.get_child("monitor", ptree()));

        boost::shared_ptr<early_stopper> es;
        boost::shared_ptr<record_optimal_training_loss> rotl;
        boost::optional<const ptree&> es_cfg 
            = cfg.get_child_optional("early_stopper");
        if(es_cfg){
            es = get_early_stopper(*m_gd, *m_mon, *es_cfg);
            if(es) {
                register_validation_batchsize(m, *es, cfg);
                rotl.reset(new record_optimal_training_loss(*es, *m_mon));
                m_mon->register_gd(*m_gd, *es);
            }else{
                m_mon->register_gd(*m_gd);
            }
        }else{
            m_mon->register_gd(*m_gd);
        }

        boost::optional<const ptree&> nc_cfg
            = cfg.get_child_optional("netcom");
        boost::shared_ptr<cuvnet::network_communication::client> client;
        boost::shared_ptr<cuvnet::network_communication::param_synchronizer> paramsync;
        if(nc_cfg){
            LOG4CXX_WARN(g_log_learner2, "Setting up netcom " <<nc_cfg->get<std::string>("host") << "/"
                    << nc_cfg->get<std::string>("db")<<"/"
                    << nc_cfg->get<std::string>("key") << "("
                    << nc_cfg->get<std::string>("push_steps") << " push, "
                    << nc_cfg->get<std::string>("pull_steps") << " pull)"
                    );
            client = boost::make_shared<cuvnet::network_communication::client>(
                    nc_cfg->get<std::string>("host"), 
                    nc_cfg->get<std::string>("db"),
                    nc_cfg->get<std::string>("key"),
                    uuid);
            paramsync = boost::make_shared<cuvnet::network_communication::param_synchronizer>(
                    "stage", *client, 
                    nc_cfg->get<int>("push_steps"),
                    nc_cfg->get<int>("pull_steps"),
                    0, 0,
                    m.get_params()
                    );
            if(es){
                ((DRGD<gradient_descent>*)m_gd.get())->set_sync_function_es(boost::ref(*paramsync), boost::ref(*es));
            }else{
                ((DRGD<gradient_descent>*)m_gd.get())->set_sync_function(boost::ref(*paramsync));
            }
        }

        bool want_wup_rec = cfg.get("gd.wup_rec.active", false);
        if(want_wup_rec){
            unsigned int wup_rec_every = cfg.get("gd.wup_rec.every", 1);
            ((WRGD<gradient_descent>*)m_gd.get())->set_monitor(*m_mon, wup_rec_every);
        }

        boost::shared_ptr<convergence_checker> cc;
        boost::optional<const ptree&> cc_cfg
            = cfg.get_child_optional("convergence_checker");
        if (cc_cfg)
            cc = get_convergence_checker(*m_gd, *m_mon, *cc_cfg);

        int time_limit = cfg.get("time_limit", INT_MAX);
        int batch_size = cfg.get("batchsize", -1);
        int max_epochs = cfg.get("gd.max_epochs", 5);
        LOG4CXX_WARN(g_log_learner2, "batchsize:"<< batch_size<< ", max_epochs:"<<max_epochs<<", time_limit:"<<time_limit);

        boost::shared_ptr<schedules::hyperparam_schedule> learnrate_schedule =
            get_learnrate_schedule(*m_gd, max_epochs, cfg.get_child("learnrate_schedule", ptree()));

        boost::shared_ptr<schedules::hyperparam_schedule> momentum_schedule =
            get_momentum_schedule(*m_gd, max_epochs, cfg.get_child("momentum_schedule", ptree()));

        bool mls_active = cfg.get("min_loss_stopper.active",false);
        boost::optional<float> target_loss;
        if(mls_active)
        { 
            target_loss = cfg.get_optional<float>("min_loss_stopper.target_loss");
        }
        
        boost::shared_ptr<stop_when_target_loss_reached> swtlr;
        if(target_loss){
            swtlr.reset(new stop_when_target_loss_reached(*m_gd, *m_mon, *target_loss));
            LOG4CXX_WARN(g_log_learner2,  "Setting up Min Loss Stopper (active:" << mls_active << ", target_loss:" << target_loss << ")");
        }
        this->before_learning(&m, *m_gd, es.get(), cfg);

        m_gd->get_swiper().dump((tmppath / "loss.dot").string(), cfg.get("verbose", false));

        SignalTranslator<CtrlCPressed> sigint(boost::bind(&gradient_descent::request_stop, m_gd.get()));
        if(batch_size < 0){
            _load_batch(&m, 0, 0);
            m_gd->batch_learning(max_epochs, time_limit);
        }else {
            m_gd->before_batch.connect(boost::bind(&learner2::_load_batch, this, &m, _1, _2));
            m_gd->current_batch_num = boost::bind(&learner2::_n_batches, this, batch_size);
            m_gd->minibatch_learning(max_epochs, time_limit);
        }
        ptree result;
        result.put("gd.result_epoch", m_gd->iters());
        result.put("gd.stop_reason", (int)m_gd->stop_reason());
        result.put("path", tmppath.string());
        if(learnrate_schedule)
            result.put("gd.final_learnrate", m_gd->learnrate());
        if(momentum_schedule)
            result.put("gd.final_momentum", 
                    boost::dynamic_pointer_cast<momentum_gradient_descent>(m_gd)->momentum());
        if(es){
            result.put("gd.early_stopper.best_perf", es->best_perf());
            if(rotl)
                result.put("gd.early_stopper.optimal_training_error", rotl->best_training_loss);
        }else{
            if(m_mon->has("loss"))
                result.put("loss", m_mon->mean("loss"));
            if(m_mon->has("cerr"))
                result.put("cerr", m_mon->mean("cerr"));
        }

        // TODO continue_learning_until_previous_loss_reached would be easier
        // if we would just not reset m_gd, and instead just run fit() again!
        m_gd.reset();
        return result;
    }

    struct save_stage_dataset{
        std::string m_path;
        std::string m_dataset;
        std::string m_stagename;
        monitor& m_monitor;
        std::ofstream m_ofs;
        boost::archive::binary_oarchive m_oa;
        unsigned int m_n_outputs;
        unsigned int m_n_batches;

        save_stage_dataset(monitor& mon, const std::string& path, const std::string& dataset, const std::string& stage, unsigned int n_outputs)
            : m_path(path), m_dataset(dataset), m_stagename(stage), m_monitor(mon),
            m_ofs(path + "/" + dataset + "-" + stage + ".ser"),
            m_oa(m_ofs, boost::archive::no_tracking),
            m_n_outputs(n_outputs),
            m_n_batches(0)
        {
        }

        void operator()(unsigned int epoch, unsigned int batch)
        {
            for(unsigned int i = 0; i < m_n_outputs; i++){
                host_matrix hm(m_monitor["output-"+boost::lexical_cast<std::string>(i)]);
                m_oa << hm;
            }
            m_n_batches ++;
        }

        ~save_stage_dataset(){
            LOG4CXX_WARN(g_log_mslearner, "save_stage_dataset: saved "<<m_n_batches<<" batches of stage `"
                    <<m_stagename<<"' to `"<<(m_path + "/"+ m_dataset + "-" + m_stagename + ".ser"));
        }
    };

    multistage_dataset::multistage_dataset(
                const std::vector<std::vector<host_matrix> >& traindata,
                const std::vector<std::vector<host_matrix> >& valdata
                ){
        m_data.reserve(traindata.size() + valdata.size());
        m_data.insert(m_data.end(), traindata.begin(), traindata.end());
        m_data.insert(m_data.end(), valdata.begin(), valdata.end());
    }
    multistage_dataset::multistage_dataset(const std::string& path, 
            const std::string& dataset,
            const std::string& stage,
            std::vector<boost::shared_ptr<ParameterInput> > inputs)
        : m_path(path), m_dataset(dataset), m_stagename(stage), m_data(inputs.size()), m_inputs(inputs)
    {
        std::string filename = path + "/" + dataset + "-" + stage + ".ser";
        std::ifstream ifs(filename.c_str());
        boost::archive::binary_iarchive ia(ifs);
        bool stop = false;
        unsigned int n_batches = 0;
        do{
            for(unsigned int i=0; i < m_inputs.size(); i++){
                host_matrix tmp;
                try{
                    ia >> tmp;
                }catch(boost::archive::archive_exception const& e){
                    stop = true;
                    break;
                }
                m_data[i].push_back(tmp);
            }
            if(!stop)
                n_batches ++;
        }while(!stop);
        LOG4CXX_WARN(g_log_mslearner, " loaded " << n_batches<<" batches from `"<<filename<<"'");
    }

    void multistage_dataset::load_batch(unsigned int epoch, unsigned int batch){
        for(unsigned int i=0; i < m_inputs.size(); i++){
            m_inputs[i]->data() = m_data[i][batch];
        }
    }

    unsigned int multistage_dataset::n_batches()const{
        return m_data[0].size();
    }

    ptree 
    multistage_learner::fit(model& _m, const ptree& cfg){

        std::vector<boost::shared_ptr<graph_modifiers::substitute_op_with_input> > saved;
        typedef models::multistage_model multistage_model;
        multistage_model& m = *dynamic_cast<multistage_model*>(&_m);
        if(&m == NULL)
            throw std::runtime_error("Need multi-stage model for multi-stage learner!");

        typedef multistage_model::stage_type stage_type;
        stage_type n_stages = m.n_stages();
        LOG4CXX_WARN(g_log_mslearner, "fitting a model with "<<n_stages<<" stages");

        cv_mode train_ds = (cv_mode)cfg.get("train_dataset", (int)CM_TRAIN);

        ptree stageres;
        std::string stage_name;
        for(unsigned int stage = 0; stage < n_stages; stage++){
            m.switch_stage(stage);
            // We need to select a dataset here, since it depends on the
            // current stage
            _switch_dataset(train_ds, -1);
            // take params from a subtree named like the stage if it exists
            stage_name = "stage_" + boost::lexical_cast<std::string>(stage);
            log4cxx::NDC ndc(stage_name);
            LOG4CXX_WARN(g_log_mslearner, "fitting `"<<stage_name<<"'");
            ptree stagecfg = cfg.get_child(stage_name, cfg);
            ptree res = learner2::fit(m, stagecfg);
            stageres.put_child(stage_name, res);
            if(stage < n_stages - 1){
                // generate new dataset using current model
                std::string next_stage_name = "stage_" + boost::lexical_cast<std::string>(stage + 1);
                ptree next_stagecfg = cfg.get_child(next_stage_name, cfg);
                if(next_stagecfg.get("switch_stage_with_outputs", false))
                {
                    int batch_size = stagecfg.get("batchsize", -1);
                    saved = switch_stage_with_outputs(m, stage, res.get<std::string>("path"), batch_size);
                }
                else{
                    saved.clear(); // repairs graph to original
                    m_current_dataset.reset();
                    BOOST_FOREACH(auto& a, m_stage_datasets){
                        a.reset();
                    }
                }
            }
        }
        saved.clear();
        for(unsigned int i=0;i<m_stage_datasets.size(); i++)
            m_stage_datasets[i].reset();
        m_current_dataset.reset();
        //m.switch_stage(0); // back to the start
        // result is result of the last stage, with the list of all stage
        // results as a subtree
        ptree result = stageres.get_child(stage_name);
        result.put_child("stage_results", stageres);
        return result;
    }

    std::vector<boost::shared_ptr<graph_modifiers::substitute_op_with_input> >
    multistage_learner::switch_stage_with_outputs(multistage_model& m,
            const multistage_model::stage_type& current_stage, const
            std::string& path, int batch_size){
        // 1. record all "outputs" of the model to a file
        std::vector<Op*> outputs = m.get_outputs();
        monitor mon;
        for(unsigned int i=0; i<outputs.size(); i++)
            mon.add(monitor::WP_SINK, outputs[i]->shared_from_this(), "output-" + boost::lexical_cast<std::string>(i));

        for(unsigned int v = 0; v < 3; v++){
            _switch_dataset(g_cm[v], -1);

            std::string stage_name = "stage_" + boost::lexical_cast<std::string>(current_stage);

            gradient_descent gd(outputs[0]->shared_from_this(), 0, std::vector<Op*>(), 0.0, 0.0);
            save_stage_dataset ssd(mon, path,
                    boost::lexical_cast<std::string>(v), stage_name,
                    outputs.size());

            gd.after_batch.connect(boost::ref(ssd));

            mon.register_gd(gd);
            if(batch_size < 0){
                _load_batch(&m, 0, 0);
                gd.batch_learning(1, INT_MAX);
            }else {
                gd.before_batch.connect(boost::bind(&learner2::_load_batch, this, &m, _1, _2));
                gd.current_batch_num = boost::bind(&learner2::_n_batches, this, batch_size);
                gd.minibatch_learning(1, INT_MAX, false); // don't shuffle
            }
        }

        // 3. substitute all "outputs" with inputs for the next stage
        std::vector<boost::shared_ptr<graph_modifiers::substitute_op_with_input> > undo;
        std::vector<boost::shared_ptr<ParameterInput> > inputs;
        for (unsigned int i = 0; i < outputs.size(); ++i)
        {
            undo.push_back(boost::make_shared<graph_modifiers::substitute_op_with_input>(
                        outputs[i]->shared_from_this()));
            inputs.push_back(undo.back()->m_input);
        }
        m_stage_datasets.clear();
        m_stage_datasets.resize(4);
        for(unsigned int v = 0; v < 3; v++){
            std::string stage_name = "stage_" + boost::lexical_cast<std::string>(current_stage);
            boost::shared_ptr<multistage_dataset> msd =
                boost::make_shared<multistage_dataset>(path,
                        boost::lexical_cast<std::string>(v), stage_name, 
                        inputs);
            m_stage_datasets[g_cm[v]] = msd;
        }
        m_stage_datasets[CM_TRAINALL] = boost::make_shared<multistage_dataset>(
                m_stage_datasets[CM_TRAIN]->m_data,
                m_stage_datasets[CM_VALID]->m_data);
        return undo;
    }

    void 
    multistage_learner::_load_batch(model* m, unsigned int epoch, unsigned int batch){
        if(!m_current_dataset)
            load_batch(m, epoch, batch);
        else
            m_current_dataset->load_batch(epoch, batch);
    }

    void
    multistage_learner::_switch_dataset(cv_mode mode, int split){
        std::string ds;
        switch(mode){
            case CM_TRAIN: ds = "TRAIN"; break;
            case CM_VALID: ds = "VALID"; break;
            case CM_TRAINALL: ds = "TRAINALL"; break;
            case CM_TEST: ds = "TEST"; break;
        }
        if(!m_stage_datasets[mode]){
            //LOG4CXX_WARN(g_log_mslearner, "_switch_dataset ORIG `"<<ds<<"':"<<split);
            //LOG4CXX_WARN(g_log, "switch_dataset virtual: mode:"<<mode<<" split:"<<split);
            switch_dataset(mode, split);
        }
        else{
            //LOG4CXX_WARN(g_log_mslearner, "_switch_dataset STAGE `"<<ds<<"':"<<split);
            m_current_dataset = m_stage_datasets[mode];
        }
    }

    unsigned int
        multistage_learner::_n_batches(unsigned int batchsize){
            if(m_current_dataset)
                return m_current_dataset->n_batches();
            return n_batches(batchsize);
        }

    ptree 
    multistage_learner::learn_until_previous_loss_reached(model& _m, const ptree& cfg, const ptree& result){
        typedef models::multistage_model multistage_model;
        multistage_model& m = *dynamic_cast<multistage_model*>(&_m);
        if(&m == NULL)
            throw std::runtime_error("Need multi-stage model for multi-stage learner!");

        typedef multistage_model::stage_type stage_type;
        stage_type n_stages = m.n_stages();

        ptree cfg2;
        for(unsigned int stage = 0; stage < n_stages; stage++){
            std::string stage_name = "stage_" + boost::lexical_cast<std::string>(stage);
            float target_loss = result.get<float>(
                "stage_results."
                +stage_name
                +".gd.early_stopper.optimal_training_error");
            ptree stagecfg = cfg.get_child(stage_name, cfg);
            LOG4CXX_WARN(g_log_mslearner, "learn_until_previous_loss_reached "<<stage_name<<", target_loss:"<<target_loss);
            stagecfg.put("early_stopper.active", false);
            stagecfg.put("min_loss_stopper.active", true);
            stagecfg.put("min_loss_stopper.target_loss", target_loss);
            cfg2.add_child(stage_name, stagecfg);
        }
        cfg2.put("batchsize", cfg.get("batchsize", -1));
        cfg2.put("train_dataset", cfg.get("train_dataset", (int)CM_TRAIN));
        ptree retrain_all_res = fit(m, cfg2);

        LOG4CXX_WARN(g_log_mslearner, "learn_until_previous_loss_reached DONE");
        return retrain_all_res;
    }
}
