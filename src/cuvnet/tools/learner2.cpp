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
                bool oserializer<binary_oarchive, cuv::tensor<float, cuv::dev_memory_space> >::tracking(const unsigned int f /* flags */) const {
                    return !(f & no_tracking);
                }
            template<>
                bool oserializer<binary_oarchive, cuv::tensor<float, cuv::host_memory_space> >::tracking(const unsigned int f /* flags */) const {
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
            LOG4CXX_INFO(g_log_learner2, "Setting up exponential learnrate schedule (initial: "<<initial
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
         * div_learnrate_schedule
         *****************************************/
        div_learnrate_schedule::div_learnrate_schedule(gradient_descent* _gd, float begin, float anneal)
            :initial(begin), anneal_start(anneal), gd(_gd)
        {
            LOG4CXX_WARN(g_log_learner2, "Setting up bergstra learnrate schedule (initial: "<<initial
                    << ", anneal_start:" << anneal_start<<")");
            con = gd->before_epoch.connect(boost::ref(*this));
        }
        void div_learnrate_schedule::operator()(unsigned int epoch, unsigned int wups)
        {
            gd->set_learnrate(
                    initial * std::min(1.0f, anneal_start / (epoch +1)));
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
    learner2::get_gradient_descent(model& m, const msg::Fit& cfg){
        const msg::GradientDescent& gdcfg = cfg.gd();
        bool wrec = gdcfg.HasExtension(msg::wup_rec_ext);
        bool drec = gdcfg.HasExtension(msg::wdiff_rec_ext);
        if(drec){
            LOG4CXX_INFO(g_log_learner2, "Setting up delta recording for gradient_descent");
        }
        if(wrec){
            LOG4CXX_INFO(g_log_learner2, "Setting up wup recording for gradient_descent");
        }
        float initial_learnrate = gdcfg.learnrate();
        float l2decay = gdcfg.l2decay();
        int verbosity = gdcfg.verbosity();
        unsigned int start_epoch = gdcfg.start_epoch();
        float initial_momentum = 0.;
        if(0){
        }else if(gdcfg.HasExtension(msg::rmsprop_ext)){
            float delta = gdcfg.GetExtension(msg::rmsprop_ext).delta();
            float grad_avg = gdcfg.GetExtension(msg::rmsprop_ext).grad_avg();
            float l1decay = gdcfg.GetExtension(msg::rmsprop_ext).l1decay();
            LOG4CXX_INFO(g_log_learner2, "Creating RMSProp gdcfg (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", grad_avg:" << grad_avg << ", delta:" << delta << ")");
            MAKE_GD(rmsprop_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay, delta, grad_avg, l1decay);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(gdcfg.HasExtension(msg::rmsprop_ext)){
            float eta_p = gdcfg.GetExtension(msg::rmsprop_ext).eta_p();
            float eta_m = gdcfg.GetExtension(msg::rmsprop_ext).eta_m();
            float l1_penalty = gdcfg.GetExtension(msg::rmsprop_ext).l1decay();
            LOG4CXX_INFO(g_log_learner2, "Creating RPROP GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", l1_penalty:"<< l1_penalty << ", eta_p:" << eta_p << ", eta_m:" << eta_m <<")");
            MAKE_GD(rprop_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay, l1_penalty, eta_p, eta_m);
            gd->set_epoch(start_epoch);
            gd->set_update_every(0);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(gdcfg.HasExtension(msg::momentum_ext)){
            LOG4CXX_INFO(g_log_learner2, "Creating Momentum GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", momentum: "<< initial_momentum << ")");
            initial_momentum = gdcfg.GetExtension(msg::momentum_ext).momentum();
            MAKE_GD(momentum_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay, initial_momentum);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(gdcfg.HasExtension(msg::adagrad_ext)){
            float winsize = gdcfg.GetExtension(msg::adagrad_ext).winsize();
            float l1_penalty = gdcfg.GetExtension(msg::adagrad_ext).l1decay();
            float delta = gdcfg.GetExtension(msg::adagrad_ext).delta();
            LOG4CXX_INFO(g_log_learner2, "Creating AdaGrad GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay << ", delta:"<< delta 
                << ", winsize:" << winsize << ", l1_penalty:" << l1_penalty << ")");
            MAKE_GD(adagrad_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay,  delta, winsize, l1_penalty);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(gdcfg.HasExtension(msg::narmsprop_ext)){
            float delta    = gdcfg.GetExtension(msg::narmsprop_ext).delta();
            float grad_avg = gdcfg.GetExtension(msg::narmsprop_ext).grad_avg();
            float step_adapt = gdcfg.GetExtension(msg::narmsprop_ext).step_adapt();
            float lr_min = gdcfg.GetExtension(msg::narmsprop_ext).lr_min();
            float lr_max = gdcfg.GetExtension(msg::narmsprop_ext).lr_max();
            LOG4CXX_INFO(g_log_learner2, "Creating NA_RMSPROP GD (initial_learnrate:" << initial_learnrate << ", momentum:" << initial_momentum <<", l2decay:" << l2decay << ", delta:" << delta << ", grad_avg:" << grad_avg << ", step_adapt:" << step_adapt << ", lr_min:" << lr_min << ", lr_max:" << lr_max << ")");
            MAKE_GD(na_rmsprop_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay,  initial_momentum, grad_avg,step_adapt,delta,lr_max,lr_min);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }else if(gdcfg.HasExtension(msg::rrmsprop_ext)){
            const msg::RRMSProp& msg = gdcfg.GetExtension(msg::rrmsprop_ext);
            float delta = msg.delta();
            float grad_avg = msg.grad_avg();
            float eta_p = msg.eta_p();
            float eta_m = msg.eta_m();
            float delta_min = msg.delta_min();
            float delta_max = msg.delta_max();
            float l1_penalty = msg.l1decay();
            LOG4CXX_INFO(g_log_learner2, "Creating RRMSPROP GD (initial_learnrate:" << initial_learnrate << ", l2decay:" << l2decay << ", l1_penalty:"<< l1_penalty <<", delta:" << delta << ", grad_avg:" << grad_avg <<  ", eta_p:" << eta_p << ", eta_m:" << eta_m << ", delta_min:" << delta_min << ", delta_max:" << delta_max << ")");
            MAKE_GD(rrmsprop_gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay, l1_penalty, grad_avg, delta/*, eta_p, eta_m, delta_max, delta_min*/);
            gd->set_epoch(start_epoch);
            gd->set_update_every(0);
            gd->set_verbosity(verbosity);
            return gd;

        }
        else{
            LOG4CXX_INFO(g_log_learner2, "Creating Plain GD (initial_learnrate:"<<initial_learnrate<<", l2decay:"<< l2decay <<")");
            MAKE_GD(gradient_descent, m.loss(), 0, m.get_params(), initial_learnrate, l2decay);
            gd->set_epoch(start_epoch);
            gd->set_verbosity(verbosity);
            return gd;
        }
    }

    boost::shared_ptr<early_stopper>
    learner2::get_early_stopper(model& m, gradient_descent& gd, monitor& mon, const msg::EarlyStopper& cfg){
        boost::shared_ptr<early_stopper> es;
        bool active = cfg.active();
        if(!active)
            return es;
        std::string watch = cfg.watch();
        float thresh = cfg.thresh();
        int every = cfg.every();
        float multiply = cfg.multiply();
        int boxfilter = cfg.boxfilter();
        int patience = cfg.patience();
        int max_steps = cfg.max_steps();
        float lr_fact = cfg.lr_fact();
        LOG4CXX_INFO(g_log_learner2, "Setting up Early Stopper (watch:"<<watch
                <<", thresh:"<< thresh
                << ", every: "<< every 
                << ", multiply:"<<multiply
                << ", boxfilter:" << boxfilter 
                << ", patience:" << patience
                << ", max_steps:" << max_steps
                << ", lr_fact:" << lr_fact
                << ")");
        es.reset(new early_stopper(gd, boost::bind(&monitor::mean, &mon, watch), thresh, every, multiply, boxfilter));
        es->before_early_stopping_epoch.connect(boost::bind(&learner2::_switch_dataset, this, CM_VALID, 0));
        es->before_early_stopping_epoch.connect(boost::bind(&model::set_predict_mode, &m, true));
        es->after_early_stopping_epoch.connect(boost::bind(&learner2::_switch_dataset, this, CM_TRAIN, 0));
        es->after_early_stopping_epoch.connect(boost::bind(&model::set_predict_mode, &m, false));
        if(max_steps > 0){
            es->decrease_lr(max_steps, lr_fact);
        }

        es->before_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_VALID, 0));
        es->after_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_TRAIN, 0));
        es->set_patience(patience);
        return es;
    }

    boost::shared_ptr<convergence_checker>
    learner2::get_convergence_checker(gradient_descent& gd, boost::shared_ptr<early_stopper> es, monitor& mon, const msg::ConvergenceChecker& cfg){
        boost::shared_ptr<convergence_checker> cc;
        bool active = cfg.active();
        if (!active)
            return cc;
        std::string watch = cfg.watch();
        float thresh = cfg.thresh();
        int min_wups = cfg.min_wups();
        float patience_inc_fact = cfg.patience_inc_fact();
        bool use_early_stopper = cfg.use_early_stopper();
        if(use_early_stopper)
            cc.reset(new convergence_checker(gd, *es,
                        boost::bind(&monitor::mean, &mon, watch), thresh, min_wups, patience_inc_fact));
        else
            cc.reset(new convergence_checker(gd,
                        boost::bind(&monitor::mean, &mon, watch), thresh, min_wups, patience_inc_fact));
        int max_steps = cfg.max_steps();
        float lr_fact = cfg.lr_fact();
        if(lr_fact > 0)
            cc->decrease_lr(max_steps, lr_fact);
        LOG4CXX_INFO(g_log_learner2, "Setting up Convergence Checker ("
                <<  "watch: " << watch
                << " thresh: " << thresh
                << " min_wups: " << min_wups
                << " use_early_stopper: " << use_early_stopper
                << " patience_inc_fact: " << patience_inc_fact
                << " max_steps: " << max_steps
                << " lr_fact: " << lr_fact
                << " )"
                );
        return cc;
    }

    boost::shared_ptr<monitor> 
    learner2::get_monitor(model& m, const msg::Monitor& cfg){
        bool verbose = cfg.verbose();
        boost::shared_ptr<monitor> mon;
        mon.reset(new monitor(verbose));
        mon->add(monitor::WP_SCALAR_EPOCH_STATS, m.loss(), "loss");
        mon->set_every(cfg.every());
        if(m.error())
            mon->add(monitor::WP_SCALAR_EPOCH_STATS, m.error(), "cerr");
        m.register_watches(*mon);
        return mon;
    }
    
    void
    learner2::register_validation_batchsize(model& m , gradient_descent& gd, early_stopper& es,
            const msg::GradientDescent& cfg,
            const msg::EarlyStopper& escfg){
        unsigned int bs_train = cfg.batch_size(), bs_valid;
        if(escfg.has_batch_size())
            bs_valid = escfg.batch_size();
        else
            bs_valid = bs_train;

        LOG4CXX_INFO(g_log_learner2, "Setting up batch sizes, validation: " << bs_valid << " train: "<<bs_train);

        gradient_descent* gdp = &gd;

        if(bs_train != bs_valid){
            es.before_early_stopping_epoch.connect([=](unsigned int){
                    gdp->current_batch_num = boost::bind(&learner2::_n_batches, this, bs_valid);
                    });
            es.before_early_stopping_epoch.connect(boost::bind(&model::set_batchsize, &m, bs_valid));
            es.before_early_stopping_epoch.connect(boost::bind(&gradient_descent::repair_swiper, m_gd));
            es.after_early_stopping_epoch.connect(boost::bind(&model::set_batchsize, &m, bs_train));
            es.after_early_stopping_epoch.connect(boost::bind(&gradient_descent::repair_swiper, m_gd));
            es.after_early_stopping_epoch.connect([=](unsigned int){
                    gdp->current_batch_num = boost::bind(&learner2::_n_batches, this, bs_train);
                    });
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
    learner2::before_predict(model* m, gradient_descent&, const msg::Predict&){
        m->set_predict_mode(true);
    }

    void 
    learner2::before_learning(model* m, gradient_descent&, cuvnet::early_stopper* es, const msg::Fit&){
        m->set_predict_mode(false);
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
    learner2::get_learnrate_schedule(gradient_descent& gd, int max_epochs, const msg::GradientDescent& cfg){
        boost::shared_ptr<schedules::hyperparam_schedule> hs;
        if(cfg.HasExtension(msg::linear_learnrate_schedule)){
            const msg::LinearSchedule& s = cfg.GetExtension(msg::linear_learnrate_schedule);
            float initial = s.initial() * gd.learnrate();
            float final   = s.final() * gd.learnrate();
            int   duration = s.duration();
            LOG4CXX_INFO(g_log_learner2, "Setting up linear learnrate schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration<<")");
            hs.reset(new schedules::linear_learnrate_schedule(&gd, initial, final, duration));
            return hs;
        }
        if(cfg.HasExtension(msg::exponential_learnrate_schedule)){
            const msg::ExponentialSchedule& s = cfg.GetExtension(msg::exponential_learnrate_schedule);
            float initial = s.initial() * gd.learnrate();
            float final   = s.final() * gd.learnrate();
            float t0   = s.t0();
            int   duration = s.duration();
            hs.reset(new schedules::exponential_learnrate_schedule(&gd, initial, final, duration, t0));
            return hs;
        }
        if(cfg.HasExtension(msg::div_learnrate_schedule)){
            const msg::DivSchedule& s = cfg.GetExtension(msg::div_learnrate_schedule);
			float initial = s.initial() * gd.learnrate();
			float anneal_start = s.annealstart();
			hs.reset(new schedules::div_learnrate_schedule(&gd, initial, anneal_start));
			return hs;
		}
        LOG4CXX_INFO(g_log_learner2, "No Learnrate Schedule!");
        return hs;
    }

    boost::shared_ptr<schedules::hyperparam_schedule> 
    learner2::get_momentum_schedule(gradient_descent& pgd, int max_epochs, const msg::GradientDescent& cfg){
        boost::shared_ptr<schedules::hyperparam_schedule> hs;

        momentum_gradient_descent* gd = dynamic_cast<momentum_gradient_descent*>(&pgd);
        if(!gd)
            return hs;

        if(cfg.HasExtension(msg::linear_momentum_schedule)){
            const msg::LinearSchedule& s = cfg.GetExtension(msg::linear_momentum_schedule); 
            float initial = s.initial();
            float final   = s.final();
            int   duration = s.duration();
            LOG4CXX_INFO(g_log_learner2, "Setting up linear momentum schedule (initial: "<<initial
                    << ", final:" << final
                    << ", duration:"<<duration<<")");
            hs.reset(new schedules::linear_momentum_schedule(gd, initial, final, duration));
            return hs;
        }
        LOG4CXX_INFO(g_log_learner2, "No Momentum Schedule!");
        return hs;
    }

    msg::FitResult
    learner2::continue_learning_until_previous_loss_reached(model& m, const msg::Fit& cfg, const msg::FitResult& result){
        // - stop when target loss reached
        // - set max_epochs=old_epoch PLUS new max_epochs
        // - let gradient_descent START from a certain number of epochs
        msg::Fit cfg2 = cfg;
        cfg2.mutable_gd()->mutable_stopping_criteria()->mutable_es()->set_active(false);
        cfg2.mutable_gd()->mutable_stopping_criteria()->set_target_loss(
                result.early_stopper().optimal_training_loss());
        cfg2.mutable_gd()->set_start_epoch(result.result_epoch());
        cfg2.mutable_gd()->mutable_stopping_criteria()->set_max_epochs(
                result.result_epoch() * 2);
        return fit(m, cfg2);
    }
    msg::FitResult 
    learner2::learn_until_previous_loss_reached(model& m, const msg::Fit& cfg, const msg::FitResult& result){
        float target_loss = result.early_stopper().optimal_training_loss();
        LOG4CXX_INFO(g_log_learner2, "multistage_learner: learn_until_previous_loss_reached, target_loss:"<<target_loss);
        msg::Fit cfg2 = cfg;
        cfg2.mutable_gd()->mutable_stopping_criteria()->mutable_es()->set_active(false);
        cfg2.mutable_gd()->mutable_stopping_criteria()->set_target_loss(target_loss);
        return fit(m, cfg2);
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

    msg::XValResult 
    crossvalidator2::fit(learner2& lrn, boost::shared_ptr<model> mptr, const msg::XVal& cfg){
        model& m = *mptr;
        unsigned int n_splits = lrn.n_splits();
        using namespace boost::accumulators;
        namespace ba = boost::accumulators;
        accumulator_set<double, 
            features<tag::mean, tag::median, tag::variance, tag::min> > s_valperf;
        accumulator_set<double, 
            features<tag::mean, tag::median, tag::variance, tag::min> > s_testperf;
        unsigned int best_split = 0;
        msg::XValResult result;
        for(unsigned int split = 0; split < n_splits; split ++){
            log4cxx::NDC ndc("split_"+boost::lexical_cast<std::string>(split));
            m.reset_params();
            lrn._switch_dataset(CM_TRAIN, split);
            msg::FitResult fit_res = lrn.fit(m, cfg.fit());
            float validation_loss = fit_res.early_stopper().best_validation_loss();
            if(split == 0 || ba::min(s_valperf) > validation_loss){
                std::string bestfile =  "after_train.ser";
                lrn.save_model(mptr, bestfile);
                best_split = split;
            }


            lrn._switch_dataset(CM_VALID);
            msg::PredictResult val_result;
            val_result.set_cerr_mean(validation_loss);

            msg::XValResult::FoldResult fold_result;
            *fold_result.mutable_fit_result() = fit_res;
            *fold_result.mutable_val_result() = val_result;

            if(cfg.evaluate_folds_on_test()){
                // evaluate on test set, (DO NOT USE for model selection!)
                lrn._switch_dataset(CM_TEST);
                msg::PredictResult test_result = lrn.predict(m, cfg.predict());
                *fold_result.mutable_test_result() = test_result;

                if(test_result.has_cerr_mean())
                    s_testperf(test_result.cerr_mean()); 
                else
                    s_testperf(test_result.loss_mean()); 
            }

            // record fold result
            *result.add_fold() = fold_result;
            s_valperf(validation_loss); 

            // TODO stop evaluating if too bad in comparison with current best?
        }
        result.set_val_mean( ba::mean(s_valperf));
        result.set_val_var(n_splits == 1 ? 0.01 : ba::variance(s_valperf));
        result.set_val_mean( ba::mean(s_testperf));
        result.set_val_mean( ba::variance(s_testperf));
        result.set_best_fold(best_split);

        // retrain on all, training + validation, if required.
        if(cfg.retrain_all()){
            log4cxx::NDC ndc("retrain_all");
            float current_loss = ba::mean(s_valperf);
            if(!cfg.has_retrain_all_thresh() || current_loss < cfg.retrain_all_thresh()){
                LOG4CXX_INFO(g_log_xval2, "retrain on TRAINVAL, current_loss:"<<current_loss<<", prev_loss:"<<cfg.retrain_all_thresh());
                m.reset_params();
                msg::Fit cfg2 = cfg.fit();
                lrn._switch_dataset(CM_TRAINALL);
                msg::FitResult tres = lrn.learn_until_previous_loss_reached(m, cfg2, result.fold(best_split).fit_result());

                lrn._switch_dataset(CM_TEST);
                msg::PredictResult pres = lrn.predict(m, cfg.predict());
                *result.mutable_retrain_all_train() = tres;
                *result.mutable_retrain_all_test() = pres;
            }else{
                LOG4CXX_INFO(g_log_xval2, "NO retrain on TRAINVAL, current_loss:"<<current_loss<<", prev_loss:"<<cfg.retrain_all_thresh());
            }
        }

        return result;
    }

    msg::PredictResult learner2::predict(model& m, const msg::Predict& cfg){
        m_mon
            = get_monitor(m, cfg.monitor());

        int batch_size = cfg.batch_size();
        gradient_descent gd(m.loss(), 0, std::vector<Op*>(), 0.0, 0.0);
        this->before_predict(&m, gd, cfg);
        m_mon->register_gd(gd);
        if(batch_size < 0){
            _load_batch(&m, 0, 0);
            gd.batch_learning(1, INT_MAX);
        }else {
            m.set_batchsize(batch_size);
            gd.repair_swiper();
            gd.before_batch.connect(boost::bind(&learner2::_load_batch, this, &m, _1, _2));
            gd.current_batch_num = boost::bind(&learner2::_n_batches, this, batch_size);
            gd.minibatch_learning(1, INT_MAX, false); // don't shuffle
        }
        msg::PredictResult result;
        if(m_mon->has("cerr")){
            result.set_cerr_mean(m_mon->mean("cerr"));
            result.set_cerr_var(m_mon->var("cerr"));
        }else{
            result.set_loss_mean(m_mon->mean("loss"));
            result.set_loss_var(m_mon->var("loss"));
        }
        return result;
    }

    msg::FitResult 
    learner2::fit(model& m, const msg::Fit& cfg)
    {
        m_gd = get_gradient_descent(m, cfg);
        m_mon = get_monitor(m, cfg.monitor());

        boost::shared_ptr<early_stopper> es;
        boost::shared_ptr<record_optimal_training_loss> rotl;
        if(cfg.gd().stopping_criteria().has_es()){
            es = get_early_stopper(m, *m_gd, *m_mon, cfg.gd().stopping_criteria().es());
            if(es) {
                register_validation_batchsize(m, *m_gd, *es, cfg.gd(), cfg.gd().stopping_criteria().es());
                rotl.reset(new record_optimal_training_loss(*es, *m_mon));
                m_mon->register_gd(*m_gd, *es);
            }else{
                m_mon->register_gd(*m_gd);
            }
        }else{
            m_mon->register_gd(*m_gd);
        }

        std::string uuid = boost::lexical_cast<std::string>(boost::uuids::uuid(boost::uuids::random_generator()()));
        boost::shared_ptr<cuvnet::network_communication::client> client;
        boost::shared_ptr<cuvnet::network_communication::param_synchronizer> paramsync;
        if(cfg.has_netcom()){
            const msg::NetCom& ncfg = cfg.netcom();
            LOG4CXX_INFO(g_log_learner2, "Setting up netcom " <<ncfg.host() << "/"
                    << ncfg.db()<<"/"
                    << ncfg.key() << "("
                    << ncfg.push_steps() << " push, "
                    << ncfg.pull_steps() << " pull)"
                    );
            client = boost::make_shared<cuvnet::network_communication::client>(
                    ncfg.host(), 
                    ncfg.db(),
                    ncfg.key(),
                    uuid);
            paramsync = boost::make_shared<cuvnet::network_communication::param_synchronizer>(
                    "stage", *client, 
                    ncfg.push_steps(),
                    ncfg.pull_steps(),
                    0, 0,
                    m.get_params()
                    );
            if(es){
                ((DRGD<gradient_descent>*)m_gd.get())->set_sync_function_es(boost::ref(*paramsync), boost::ref(*es));
            }else{
                ((DRGD<gradient_descent>*)m_gd.get())->set_sync_function(boost::ref(*paramsync));
            }
        }

        bool want_wup_rec = cfg.gd().HasExtension(msg::wup_rec_ext);
        if(want_wup_rec){
            unsigned int wup_rec_every = cfg.gd().GetExtension(msg::wup_rec_ext).every();
            ((WRGD<gradient_descent>*)m_gd.get())->set_monitor(*m_mon, wup_rec_every);
        }

        boost::shared_ptr<convergence_checker> cc;
        if (cfg.gd().stopping_criteria().has_cc())
            cc = get_convergence_checker(*m_gd, es, *m_mon, cfg.gd().stopping_criteria().cc());

        int time_limit = cfg.gd().stopping_criteria().time_limit();
        int max_epochs = cfg.gd().stopping_criteria().max_epochs();
        int batch_size = cfg.gd().batch_size();
        LOG4CXX_INFO(g_log_learner2, "batchsize:"<< batch_size<< ", max_epochs:"<<max_epochs<<", time_limit:"<<time_limit);

        boost::shared_ptr<schedules::hyperparam_schedule> learnrate_schedule =
            get_learnrate_schedule(*m_gd, max_epochs, cfg.gd());

        boost::shared_ptr<schedules::hyperparam_schedule> momentum_schedule =
            get_momentum_schedule(*m_gd, max_epochs, cfg.gd());

        float target_loss = cfg.gd().stopping_criteria().target_loss();
        boost::shared_ptr<stop_when_target_loss_reached> swtlr;
        if(target_loss > 0.){
            swtlr.reset(new stop_when_target_loss_reached(*m_gd, *m_mon, target_loss));
            LOG4CXX_INFO(g_log_learner2,  "Setting up Min Loss Stopper (target_loss:" << target_loss << ")");
        }

        m_gd->get_swiper().dump("loss.dot", false);
        m_gd->get_swiper().dump("loss-verbose.dot", true);

        SignalTranslator<CtrlCPressed> sigint(boost::bind(&gradient_descent::request_stop, m_gd.get()));
        if(batch_size < 0){
            _load_batch(&m, 0, 0);
            this->before_learning(&m, *m_gd, es.get(), cfg);
            m_gd->batch_learning(max_epochs, time_limit);
        }else {
            m_gd->before_batch.connect(boost::bind(&learner2::_load_batch, this, &m, _1, _2));
            m_gd->current_batch_num = boost::bind(&learner2::_n_batches, this, batch_size);
            this->before_learning(&m, *m_gd, es.get(), cfg);
            m_gd->minibatch_learning(max_epochs, time_limit);
        }
        msg::FitResult result;
        result.set_result_epoch(m_gd->iters());
        result.set_stop_reason((msg::FitResult_StopReason)m_gd->stop_reason());
        if(learnrate_schedule)
            result.set_final_learnrate(m_gd->learnrate());
        if(momentum_schedule)
            result.set_final_momentum(
                    boost::dynamic_pointer_cast<momentum_gradient_descent>(m_gd)->momentum());
        if(es){
            msg::EarlyStopperResult esr;
            esr.set_best_validation_loss(es->best_perf());
            if(rotl)
                esr.set_optimal_training_loss(rotl->best_training_loss);
            m_es = es; // contains infos that may be valuable to cross-validation etc.
            *result.mutable_early_stopper() = esr;
        }else{
            if(m_mon->has("loss"))
                result.set_loss(m_mon->mean("loss"));
            if(m_mon->has("cerr"))
                result.set_cerr(m_mon->mean("cerr"));
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
            LOG4CXX_INFO(g_log_mslearner, "save_stage_dataset: saved "<<m_n_batches<<" batches of stage `"
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
        LOG4CXX_INFO(g_log_mslearner, " loaded " << n_batches<<" batches from `"<<filename<<"'");
    }

    void multistage_dataset::load_batch(unsigned int epoch, unsigned int batch){
        for(unsigned int i=0; i < m_inputs.size(); i++){
            m_inputs[i]->data() = m_data[i][batch];
        }
    }

    unsigned int multistage_dataset::n_batches()const{
        return m_data[0].size();
    }

    msg::FitResult 
    multistage_learner::fit(model& _m, const msg::Fit& cfg){

        std::vector<boost::shared_ptr<graph_modifiers::substitute_op_with_input> > saved;
        typedef models::multistage_model multistage_model;
        multistage_model& m = *dynamic_cast<multistage_model*>(&_m);
        if(&m == NULL)
            throw std::runtime_error("Need multi-stage model for multi-stage learner!");

        typedef multistage_model::stage_type stage_type;
        stage_type n_stages = m.n_stages();
        LOG4CXX_INFO(g_log_mslearner, "fitting a model with "<<n_stages<<" stages");

        cv_mode train_ds = this->current_cvmode();

        msg::FitResult stageres;
        std::string stage_name;
        for(unsigned int stage = 0; stage < n_stages; stage++){
            m.switch_stage(stage);
            // We need to select a dataset here, since it depends on the
            // current stage
            _switch_dataset(train_ds, -1);
            // take params from a subtree named like the stage if it exists
            stage_name = "stage_" + boost::lexical_cast<std::string>(stage);
            log4cxx::NDC ndc(stage_name);
            LOG4CXX_INFO(g_log_mslearner, "fitting `"<<stage_name<<"'");
            msg::Fit stagecfg = 
                  cfg.HasExtension(msg::multistage_ext) 
                && cfg.GetExtension(msg::multistage_ext).stage_size() > (int)stage
                ? cfg.GetExtension(msg::multistage_ext).stage(stage)
                : cfg;
            msg::FitResult res = learner2::fit(m, stagecfg);
            *stageres.add_stage() = res;
            if(stage < n_stages - 1){
                // generate new dataset using current model
                std::string next_stage_name = "stage_" + boost::lexical_cast<std::string>(stage + 1);
                msg::Fit next_stagecfg = 
                    cfg.HasExtension(msg::multistage_ext)
                    && cfg.GetExtension(msg::multistage_ext).stage_size() > (int)stage+1
                    ? cfg.GetExtension(msg::multistage_ext).stage(stage + 1)
                    : cfg;
                if(next_stagecfg.GetExtension(msg::multistage_ext).switch_stage_with_outputs())
                {
                    int batch_size = 
                        cfg.gd().stopping_criteria().has_es()
                        ? cfg.gd().stopping_criteria().es().batch_size()
                        : cfg.gd().batch_size();
                    saved = switch_stage_with_outputs(m, stage, ".", batch_size);
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
        msg::FitResult result = stageres.stage(n_stages-1);
        for(unsigned int i=0;i<n_stages; i++)
            *result.add_stage() = stageres.stage(i);
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
            //LOG4CXX_INFO(g_log_mslearner, "_switch_dataset ORIG `"<<ds<<"':"<<split);
            //LOG4CXX_INFO(g_log, "switch_dataset virtual: mode:"<<mode<<" split:"<<split);
            switch_dataset(mode, split);
        }
        else{
            //LOG4CXX_INFO(g_log_mslearner, "_switch_dataset STAGE `"<<ds<<"':"<<split);
            m_current_dataset = m_stage_datasets[mode];
        }
        m_current_cvmode = mode;
    }

    unsigned int
        multistage_learner::_n_batches(unsigned int batchsize){
            if(m_current_dataset)
                return m_current_dataset->n_batches();
            return n_batches(batchsize);
        }

    msg::FitResult 
    multistage_learner::learn_until_previous_loss_reached(model& _m, const msg::Fit& cfg, const msg::FitResult& result){
        TRACE(g_log_learner2, "luplr");
        typedef models::multistage_model multistage_model;
        multistage_model& m = *dynamic_cast<multistage_model*>(&_m);
        if(&m == NULL)
            throw std::runtime_error("Need multi-stage model for multi-stage learner!");

        typedef multistage_model::stage_type stage_type;
        stage_type n_stages = m.n_stages();

        msg::Fit cfg2;
        msg::FitResult stageres;
        for(unsigned int stage = 0; stage < n_stages; stage++){
            std::string stage_name = "stage_" + boost::lexical_cast<std::string>(stage);
            float target_loss = result.stage(stage).early_stopper().optimal_training_loss();
            msg::Fit stagecfg = 
                cfg.HasExtension(msg::multistage_ext)
                ? cfg.GetExtension(msg::multistage_ext).stage(stage)
                : cfg;
            LOG4CXX_INFO(g_log_mslearner, "learn_until_previous_loss_reached "<<stage_name<<", target_loss:"<<target_loss);
            stagecfg.mutable_gd()->mutable_stopping_criteria()->mutable_es()->set_active(false);
            stagecfg.mutable_gd()->mutable_stopping_criteria()->set_target_loss(
                    result.early_stopper().optimal_training_loss());
            stagecfg.mutable_gd()->mutable_stopping_criteria()->set_max_epochs(
                    result.result_epoch() * 2);
            *(cfg2.MutableExtension(msg::multistage_ext)->add_stage()) = stagecfg;
        }
        return fit(m, cfg2);
    }
}
