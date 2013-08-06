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
        exponential_learnrate_schedule::exponential_learnrate_schedule(gradient_descent* _gd, float begin, float end, int epochs, float t0_)
            :initial(begin), final(end), duration(epochs), t0(t0_), gd(_gd)
        {
            alpha = std::log(final/initial) / std::log(t0/(t0 + duration));
            eta0  = initial * std::pow(t0, alpha);
            LOG4CXX_WARN(g_log, "Setting up exponential learnrate schedule (initial: "<<initial
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
    learner2::get_gradient_descent(model& m, const ptree& cfg, const std::string& stage){
        std::string typ = cfg.get("type", "plain");
        float initial_learnrate = cfg.get("learnrate", 0.01);
        float initial_momentum = cfg.get("momentum", 0.9);
        float l2decay = cfg.get("l2decay", 0.0);
        unsigned int start_epoch = cfg.get("start_epoch", 0);
        if(typ == "plain"){
            LOG4CXX_WARN(g_log, "Creating Plain GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<gradient_descent>(m.loss(stage), 0, m.get_params(stage), initial_learnrate, l2decay);
            gd->set_epoch(start_epoch);
            return gd;
        }else if(typ == "rmsprop"){
            LOG4CXX_WARN(g_log, "Creating RMSProp GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            float delta = cfg.get("delta", 0.01f);
            float grad_avg = cfg.get("grad_avg", 0.9f);
            float l1decay = cfg.get("l1decay", 0.f);
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<rmsprop_gradient_descent>(m.loss(stage), 0, m.get_params(stage), initial_learnrate, l2decay, delta, grad_avg, l1decay);
            gd->set_epoch(start_epoch);
            return gd;
        }else if(typ == "rprop"){
            LOG4CXX_WARN(g_log, "Creating RPROP GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<rprop_gradient_descent>(m.loss(stage), 0, m.get_params(stage), initial_learnrate, l2decay);
            gd->set_epoch(start_epoch);
            return gd;
        }else if(typ == "momentum"){
            LOG4CXX_WARN(g_log, "Creating Momentum GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", momentum: "<< initial_momentum << ")");
            boost::shared_ptr<gradient_descent> gd =
                boost::make_shared<momentum_gradient_descent>(m.loss(stage), 0, m.get_params(stage), initial_learnrate, l2decay, initial_momentum);
            gd->set_epoch(start_epoch);
            return gd;
        }else{
            throw std::runtime_error("unknown/not implemented gd_type: " + typ);
        }
    }

    boost::shared_ptr<early_stopper>
    learner2::get_early_stopper(gradient_descent& gd, monitor& mon, const ptree& cfg, const std::string& stage){
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
        es->before_early_stopping_epoch.connect(boost::bind(&learner2::switch_dataset, this, CM_VALID, 0, stage));
        es->after_early_stopping_epoch.connect(boost::bind(&learner2::switch_dataset, this, CM_TRAIN, 0, stage));

        es->before_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_VALID, 0));
        es->after_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_TRAIN, 0));
        es->set_patience(patience);
        return es;
    }

    boost::shared_ptr<monitor> 
    learner2::get_monitor(model& m, const ptree& cfg, const std::string& stage){
        bool verbose = cfg.get("verbose", true);
        boost::shared_ptr<monitor> mon = boost::make_shared<monitor>(verbose);
        mon->add(monitor::WP_SCALAR_EPOCH_STATS, m.loss(stage), "loss");
        if(m.error())
            mon->add(monitor::WP_SCALAR_EPOCH_STATS, m.error(stage), "cerr");
        m.register_watches(*mon, stage);
        return mon;
    }

    void 
    learner2::load_batch(model* m, unsigned int batch, const std::string& stage){
    }

    void 
    learner2::before_learning(model* m, gradient_descent&, cuvnet::early_stopper* es){
    }

    unsigned int 
    learner2::n_batches(unsigned int batchsize){
        return 1;
    }

    void learner2::switch_dataset(cv_mode mode, int split, const std::string& stage){
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
    learner2::continue_learning_until_previous_loss_reached(model& m, const ptree& cfg, const ptree& result, const std::string& stage){
        // - stop when target loss reached
        // - set max_epochs=old_epoch PLUS new max_epochs
        // - let gradient_descent START from a certain number of epochs
        ptree cfg2 = cfg;
        cfg2.put("early_stopper.active", false);
        cfg2.put("min_loss_stopper.active", true);
        cfg2.put("min_loss_stopper.target_loss", result.get<float>("gd.early_stopper.optimal_training_error"));
        cfg2.put("gd.start_epoch",    result.get<unsigned int>("gd.result_epoch"));
        cfg2.put("gd.max_epochs", 2 * result.get<unsigned int>("gd.result_epoch"));
        ptree res = fit(m, cfg2, stage);
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
        std::vector<std::string> stages;
        {
            std::string tmp = cfg.get("stages", "");
                boost::split(stages, tmp, boost::is_any_of(";"));
        }
        accumulator_set<double, 
            features<tag::mean, tag::median, tag::variance, tag::min> > s_valperf;
        unsigned int best_split = 0;
        bfs::path bestfile;
        ptree best_result;
        for(unsigned int split = 0; split < n_splits; split ++){
            m.reset_params();
            std::vector<ptree> stageres;
            for(unsigned int stage = 0; stage < stages.size(); stage++){
                switch_dataset(CM_TRAIN, split, stages[stage]);
                // take params from a subtree named like the stage if it exists
                ptree res = fit(m, cfg.get_child(stages[stage], cfg), stages[stage]);
                stageres.push_back(res);
                // TODO generate new dataset using current model?
            }
            if(stages.size() == 1)
                result.add_child("xval.folds", stageres.back());
            else{
                for(unsigned int stage = 0; stage < stages.size(); stage++){
                    result.add_child("xval.folds." + stages[stage], stageres[stage]);
                }
            }


            result.put("xval.mean", ba::mean(s_valperf));
            result.put("xval.var", ba::variance(s_valperf));
            if(split == 0 || ba::min(s_valperf) > stageres.back().get<float>("gd.early_stopper.best_perf")){
                bestfile =  bfs::path(stageres.back().get<std::string>("path")) / "after_train.ser";
                save_model(m, bestfile.string());
                best_split = split;
                best_result = stageres.back();
            }
            s_valperf(stageres.back().get<float>("gd.early_stopper.best_perf")); 
            // TODO stop evaluating if too bad in comparison with current best?
        }
        result.put_child("xval.best_fold", best_result);
        return result;
    }

    ptree 
    learner2::fit(model& m, const ptree& cfg, const std::string& stage)
    {
        std::string uuid = boost::lexical_cast<std::string>(boost::uuids::uuid(boost::uuids::random_generator()()));
        bfs::path tmppath = bfs::path("experiments") / uuid;
        tmppath = cfg.get("path", tmppath.string());

        boost::shared_ptr<gradient_descent> gd 
            = get_gradient_descent(m, cfg.get_child("gd"), stage);

        boost::shared_ptr<monitor> mon 
            = get_monitor(m, cfg.get_child("monitor", ptree()), stage);

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
        LOG4CXX_WARN(g_log, "stage:`"<<stage<<"', batchsize:"<< batch_size<< ", max_epochs:"<<max_epochs<<", time_limit:"<<time_limit);

        boost::shared_ptr<schedules::hyperparam_schedule> learnrate_schedule =
            get_learnrate_schedule(*gd, max_epochs, cfg.get_child("learnrate_schedule", ptree()));

        boost::shared_ptr<schedules::hyperparam_schedule> momentum_schedule =
            get_momentum_schedule(*gd, max_epochs, cfg.get_child("momentum_schedule", ptree()));

        boost::optional<float> target_loss = 
            cfg.get_optional<float>("target_loss");
        boost::shared_ptr<stop_when_target_loss_reached> swtlr;
        if(target_loss)
            swtlr.reset(new stop_when_target_loss_reached(*gd, *mon, *target_loss));

        this->before_learning(&m, *gd, es.get());
        if(batch_size < 0){
            load_batch(&m, 0, stage);
            gd->batch_learning(max_epochs, time_limit);
        }else {
            gd->before_batch.connect(boost::bind(&learner2::load_batch, this, &m, _1, stage));
            gd->current_batch_num = boost::bind(&learner2::n_batches, this, batch_size);
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
