#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/min.hpp>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/filesystem.hpp>

#include <cuvnet/tools/serialization_helper.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <cuvnet/tools/gradient_descent.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/tools/logging.hpp>

#include "learner2.hpp"

namespace bfs = boost::filesystem;

typedef boost::property_tree::ptree ptree;

namespace 
{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("learner2"));
}

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
        exponential_learnrate_schedule::exponential_learnrate_schedule(gradient_descent* _gd, float begin, float end, int epochs)
            :initial(begin), final(end), duration(epochs), gd(_gd)
        {
            con = gd->before_epoch.connect(boost::ref(*this));
        }
        void exponential_learnrate_schedule::operator()(unsigned int epoch, unsigned int wups)
        {
            gd->set_learnrate(
                    std::max(final, 
                        initial * 
                        std::pow(
                            final/initial, 
                            duration / (float) duration)));
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
            :current_training_loss(0.f), best_training_loss(0.f), mon(&_mon){
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

    /*****************************************
     * Learner2
     *****************************************/
    boost::shared_ptr<gradient_descent> 
    learner2::get_gradient_descent(model& m, const ptree& cfg){
        std::string typ = cfg.get("type", "plain");
        float initial_learnrate = cfg.get("learnrate", 0.01);
        float initial_momentum = cfg.get("momentum", 0.9);
        float l2decay = cfg.get("l2decay", 0.0);
        unsigned int start_epoch = cfg.get("start_epoch", 0);
        if(typ == "plain"){
            LOG4CXX_WARN(g_log, "Creating Plain GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<gradient_descent>(m.loss(), 0, m.get_params(), initial_learnrate, l2decay);
            gd->set_epoch(start_epoch);
            return gd;
        }else if(typ == "rmsprop"){
            LOG4CXX_WARN(g_log, "Creating RMSProp GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            float delta = cfg.get("delta", 0.01f);
            float grad_avg = cfg.get("grad_avg", 0.9f);
            float l1decay = cfg.get("l1decay", 0.f);
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<rmsprop_gradient_descent>(m.loss(), 0, m.get_params(), initial_learnrate, l2decay, delta, grad_avg, l1decay);
            gd->set_epoch(start_epoch);
            return gd;
        }else if(typ == "rprop"){
            LOG4CXX_WARN(g_log, "Creating RPROP GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<rprop_gradient_descent>(m.loss(), 0, m.get_params(), initial_learnrate, l2decay);
            gd->set_epoch(start_epoch);
            return gd;
        }else if(typ == "momentum"){
            LOG4CXX_WARN(g_log, "Creating Momentum GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", momentum: "<< initial_momentum << ")");
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<momentum_gradient_descent>(m.loss(), 0, m.get_params(), initial_learnrate, l2decay, initial_momentum);
            gd->set_epoch(start_epoch);
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
        LOG4CXX_WARN(g_log, "Setting up Early Stopper (watch:"<<watch
                <<", thresh:"<< thresh
                << ", every: "<< every 
                << ", multiply:"<<multiply
                << ", boxfilter:" << boxfilter 
                << ", patience:" << patience<< ")");
        es.reset(new early_stopper(gd, boost::bind(&monitor::mean, &mon, watch), thresh, every, multiply, boxfilter));
        es->before_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_VALID, 0));
        es->after_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_TRAIN, 0));
        es->set_patience(patience);
        return es;
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
    learner2::load_batch(model* m, unsigned int batch){
    }

    unsigned int 
    learner2::n_batches(){
        return 1;
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
            LOG4CXX_WARN(g_log, "Setting up linear learnrate schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration<<")");
            hs.reset(new schedules::linear_learnrate_schedule(&gd, initial, final, duration));
            return hs;
        }
        if(schedule == "exponential"){
            float initial = cfg.get("initial", gd.learnrate());
            float final   = cfg.get("final", 0.000001f);
            int   duration = cfg.get("duration", max_epochs);
            LOG4CXX_WARN(g_log, "Setting up exponential learnrate schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration<<")");
            hs.reset(new schedules::exponential_learnrate_schedule(&gd, initial, final, duration));
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
            LOG4CXX_WARN(g_log, "Setting up linear momentum schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration<<")");
            hs.reset(new schedules::linear_momentum_schedule(gd, initial, final, duration));
            return hs;
        }
        throw std::runtime_error("unknown learnrate schedule: " + schedule);
    }

    void 
    learner2::save_model(model& m, std::string filename){
        namespace bar= boost::archive;
        std::ofstream f(filename.c_str());
        bar::binary_oarchive oa(f);
        register_objects(oa);
        oa << m;
    }
    void 
    learner2::load_model(model& m, std::string filename){
        namespace bar= boost::archive;
        std::ifstream f(filename.c_str());
        bar::binary_iarchive ia(f);
        register_objects(ia);
        ia >> m;
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
        cfg2.put("gd.max_epochs", 2 * result.get<unsigned int>("gd.result_epoch"));
        ptree res = fit(m, cfg2);
        return res;
    }

    unsigned int learner2::n_splits()const{
        return 1;
    }

    ptree 
    learner2::crossvalidation_fit(model& m, const ptree& cfg){
        unsigned int n_splits = this->n_splits();
        ptree result;
        using namespace boost::accumulators;
        namespace ba = boost::accumulators;
        accumulator_set<double, 
            features<tag::mean, tag::median, tag::variance, tag::min> > s_valperf;
        unsigned int best_split = 0;
        bfs::path bestfile;
        ptree best_result;
        for(unsigned int split = 0; split < n_splits; split ++){
            m.reset_params();
            switch_dataset(CM_TRAIN, split);
            ptree res = fit(m, cfg);
            result.add_child("xval.folds", res);
            result.put("xval.mean", ba::mean(s_valperf));
            result.put("xval.var", ba::variance(s_valperf));
            if(split == 0 || ba::min(s_valperf) < res.get<float>("gd.early_stopper.best_perf")){
                bestfile =  bfs::path(res.get<std::string>("path")) / "after_train.ser";
                save_model(m, bestfile.string());
                best_split = split;
                best_result = res;
            }
            // TODO stop evaluating if too bad in comparison with current best?
        }
        result.put_child("xval.best_fold", best_result);
        return result;
    }

    ptree 
    learner2::fit(model& m, const ptree& cfg)
    {
        std::string uuid = boost::lexical_cast<std::string>(boost::uuids::uuid(boost::uuids::random_generator()()));
        bfs::path tmppath = bfs::path("experiments") / uuid;
        tmppath = cfg.get("path", tmppath.string());

        boost::shared_ptr<gradient_descent> gd 
            = get_gradient_descent(m, cfg.get_child("gd"));

        boost::shared_ptr<monitor> mon 
            = get_monitor(m, cfg.get_child("monitor", ptree()));

        boost::shared_ptr<early_stopper> es;
        boost::shared_ptr<record_optimal_training_loss> rotl;
        boost::optional<const ptree&> es_cfg 
            = cfg.get_child_optional("early_stopper");
        if(es_cfg){
            es = get_early_stopper(*gd, *mon, *es_cfg);
            if(es) {
                rotl.reset(new record_optimal_training_loss(*es, *mon));
                mon->register_gd(*gd, *es);
            }else{
                mon->register_gd(*gd);
            }
        }else{
            mon->register_gd(*gd);
        }

        int time_limit = cfg.get("time_limit", INT_MAX);
        int batch_size = cfg.get("batchsize", -1);
        int max_epochs = cfg.get("gd.max_epochs", 5);
        LOG4CXX_WARN(g_log, "batchsize:"<< batch_size<< ", max_epochs:"<<max_epochs<<", time_limit:"<<time_limit);

        boost::shared_ptr<schedules::hyperparam_schedule> learnrate_schedule =
            get_learnrate_schedule(*gd, max_epochs, cfg.get_child("learnrate_schedule", ptree()));

        boost::shared_ptr<schedules::hyperparam_schedule> momentum_schedule =
            get_momentum_schedule(*gd, max_epochs, cfg.get_child("momentum_schedule", ptree()));

        boost::optional<float> target_loss = 
            cfg.get_optional<float>("target_loss");
        boost::shared_ptr<stop_when_target_loss_reached> swtlr;
        if(target_loss)
            swtlr.reset(new stop_when_target_loss_reached(*gd, *mon, *target_loss));

        if(batch_size < 0){
            load_batch(&m, 0);
            gd->batch_learning(max_epochs, time_limit);
        }else {
            gd->before_batch.connect(boost::bind(&learner2::load_batch, this, &m, _1));
            gd->current_batch_num = boost::bind(&learner2::n_batches, this);
            gd->minibatch_learning(max_epochs, time_limit);
        }
        ptree result;
        result.put("gd.result_epoch", gd->iters());
        result.put("path", tmppath.string());
        if(learnrate_schedule)
            result.put("gd.final_learnrate", gd->learnrate());
        if(momentum_schedule)
            result.put("gd.final_momentum", 
                    boost::dynamic_pointer_cast<momentum_gradient_descent>(gd)->momentum());
        if(es){
            result.put("gd.early_stopper.best_perf", es->best_perf());
            if(rotl)
                result.put("gd.early_stopper.optimal_training_error", rotl->best_training_loss);
        }else{
            if(mon->has("loss"))
                result.put("loss", mon->mean("loss"));
            if(mon->has("cerr"))
                result.put("cerr", mon->mean("cerr"));
        }
        return result;
    }
}
