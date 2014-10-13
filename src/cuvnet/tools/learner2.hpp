#ifndef __LEARNER2_HPP__
#define __LEARNER2_HPP__

#include <boost/signals2.hpp>
#include <cuvnet/models/models.hpp>
#include <datasets/dataset.hpp>    /* for cv_mode */
#include <cuvnet/graph_modifiers.hpp>
#include <cuvnet/messages/gd.pb.h>

namespace cuvnet
{
    class monitor;
    class gradient_descent;
    class early_stopper;
    class convergence_checker;
    class momentum_gradient_descent;

    /**
     * contains schedules for learning rates, momentums, etc.
     * @ingroup learning
     */
    namespace schedules
    {
        /**
         * An interface for hyperparameter schedules. 
         *
         * Hyperparameter schedules are updated in the before-epoch event.
         * @ingroup learning
         */
        struct hyperparam_schedule{
            virtual void operator()(unsigned int epoch, unsigned int wups)=0;
            virtual ~hyperparam_schedule(){}
        };
        /**
         * A linearly decreasing learnrate schedule.
         * @ingroup learning
         */
        struct linear_learnrate_schedule : public hyperparam_schedule{
            float initial, final, duration;
            gradient_descent* gd;
            boost::signals2::scoped_connection con;
            linear_learnrate_schedule(gradient_descent* _gd, float begin, float end, int epochs);
            virtual void operator()(unsigned int epoch, unsigned int wups);
        };
        /**
         * An exponentially decreasing learnrate schedule
         * @ingroup learning
         */
        struct exponential_learnrate_schedule : public hyperparam_schedule{
            float initial, final, duration, t0, alpha, eta0;
            gradient_descent* gd;
            boost::signals2::scoped_connection con;
            exponential_learnrate_schedule(gradient_descent* _gd, float begin, float end, int epochs, float t0);
            virtual void operator()(unsigned int epoch, unsigned int wups);
        };
        /**
         * A hyper-param-friendly learnrate schedule from this hyperopt model search:
         * https://github.com/hyperopt/hyperopt-nnet/blob/master/hpnnet/nnet.py
         * @ingroup learning
         */
        struct div_learnrate_schedule : public hyperparam_schedule{
            float initial, anneal_start;
            gradient_descent* gd;
            boost::signals2::scoped_connection con;
            div_learnrate_schedule(gradient_descent* _gd, float begin, float anneal);
            virtual void operator()(unsigned int epoch, unsigned int wups);
        };
        /**
         * A linearly increasing momentum schedule.
         * @ingroup learning
         */
        struct linear_momentum_schedule : public hyperparam_schedule{
            float initial, final, duration;
            momentum_gradient_descent* gd;
            boost::signals2::scoped_connection con;
            linear_momentum_schedule(momentum_gradient_descent* _gd, float begin, float end, int epochs);
            virtual void operator()(unsigned int epoch, unsigned int wups);
        };
    }

    /**
     * A utility class that records the training loss whenever the current validation loss improved.
     */
    struct record_optimal_training_loss{
        float current_training_loss, best_training_loss;
        monitor* mon;
        boost::signals2::scoped_connection con0, con1;
        record_optimal_training_loss(early_stopper& es, monitor& _mon);
        void before_early_stopping_epoch(unsigned int);
        void improved();
    };

    /**
     * A utility class that terminates learning whenever the current training loss is less than some given value.
     */
    struct stop_when_target_loss_reached{
        float target_loss;
        monitor* mon;
        boost::signals2::scoped_connection con;
        stop_when_target_loss_reached(gradient_descent& gd, monitor& _mon, float tloss);
        void operator()(unsigned int current_epoch, unsigned int wups);
    };

    /**
     * sets up common learning features and fits given models.
     * @ingroup learning
     */
    class learner2{
        private:
            boost::shared_ptr<gradient_descent> m_gd;
            boost::shared_ptr<early_stopper> m_es;
            boost::shared_ptr<monitor> m_mon;
        protected:
            typedef models::model model;

            /**
             * Overload this to load a specific batch from your dataset into the model.
             *
             * Does nothing by default.
             *
             * @param m the model to load the batch into
             * @param epoch current epoch number
             * @param bid the batch id which is to be loaded into the model
             */
            virtual void load_batch(model* m, unsigned int epoch, unsigned int bid);

            /**
             * In this hook you can modify the gradient_descent object just before learning.
             * 
             * Does nothing by default.
             */
            virtual void before_learning(model* m, gradient_descent& gd, cuvnet::early_stopper* es, const msg::Fit& cfg);

            /**
             * In this hook you can modify the gradient_descent object just before calling predict.
             * 
             * Does nothing by default.
             */
            virtual void before_predict(model* m, gradient_descent& gd, const msg::Predict& cfg);

            /**
             * Returns a gradient Gradient Descent object described by the configuration parameter.
             *
             * Not all gradient_descent objects in cuvnet are currently
             * supported, but it should be easy to add them, either by
             * modifying this function or by overloading it.
             *
             * @param m the model for which to generate the gd object
             * @param cfg the configuration subtree describing how the gd object should be created
             */
            virtual
            boost::shared_ptr<gradient_descent> 
                get_gradient_descent(model& m, const msg::Fit& cfg);

            /**
             * Returns an early stopper or NULL, configured according to the cfg parameter.
             */
            boost::shared_ptr<early_stopper>
                get_early_stopper(model& m, gradient_descent& gd, monitor& mon, const msg::EarlyStopper& cfg);

            /**
             * Returns an convergence checker or NULL, configured according to the cfg parameter.
             */
            boost::shared_ptr<convergence_checker>
                get_convergence_checker(gradient_descent& gd, boost::shared_ptr<early_stopper> es, monitor& mon, const msg::ConvergenceChecker& cfg);

            /**
             * Returns a monitor that watches loss and error of the model.
             *
             * It also asks the model to register possible other watchpoints
             * with the monitor using model::register_watches().
             *
             * @param m the model that we want to monitor
             * @param cfg the configuration subtree of the monitor
             */
            boost::shared_ptr<monitor> 
                get_monitor(model& m, const msg::Monitor& cfg);

            /**
             * Returns a learnrate schedule, which will be called in the before_epoch event.
             */
            boost::shared_ptr<schedules::hyperparam_schedule> 
                virtual get_learnrate_schedule(gradient_descent& gd, int max_epochs, const msg::GradientDescent& cfg);

            /**
             * Returns a momentum schedule, which will be called in the before_epoch event.
             */
            boost::shared_ptr<schedules::hyperparam_schedule> 
                virtual get_momentum_schedule(gradient_descent& pgd, int max_epochs, const msg::GradientDescent& cfg);

        public:

            /**
             * Determine how many splits there are for crossvalidation.
             *
             * @return 1 by default
             */
            virtual unsigned int n_splits()const;

            /**
             * 
             * This function calls n_batches, and is overloaded by multistage
             * learner, and you should likely neither overload nor touch it.
             *
             * @param batchsize the number of instances in one batch
             * @return 1 by default
             */
            virtual unsigned int _n_batches(unsigned int batchsize);

            /**
             * Overload this to tell learner how many batches your dataset has.
             *
             * @param batchsize the number of instances in one batch
             * @return 1 by default
             */
            virtual unsigned int n_batches(unsigned int batchsize);

            /**
             * Serialize the model to a file. 
             * This is needed by crossvalidation to save the best intermediate model.
             *
             * @param m the model to be serialized
             * @param filename the filename where the model should end up in
             */
            void 
                save_model(boost::shared_ptr<model>& m, std::string filename);

            /**
             * Deserialize the model from a file. 
             * This is needed by crossvalidation to load the best intermediate model.
             *
             * @param m the model to be serialized
             * @param filename the filename where the model should end up in
             */
            void 
                load_model(boost::shared_ptr<model>& m, std::string filename);

            /**
             * Fit a model, using features described in configuration param.
             *
             * @param m the model to be fitted
             * @param cfg parameters for fitting
             */
            virtual msg::FitResult fit(model& m, const msg::Fit& cfg);

            /**
             * Evaluate the model for the current dataset (one
             * pass), using features described in configuration param.
             *
             * @param m the model to be evaluated
             * @param cfg parameters for evaluation
             */
            virtual msg::PredictResult predict(model& m, const msg::Predict& cfg);

            /**
             * Continue learning in an already learned model, eg on TRAINVAL instead of on TRAIN.
             *
             * @param m the model to be learned further.
             * @param cfg how to train (probably the same thing given to fit() previously)
             * @param result the result of fit(), or the best result of crossvalidation_fit
             */
            virtual msg::FitResult continue_learning_until_previous_loss_reached(model& m, const msg::Fit& cfg, const msg::FitResult& result);

            /**
             * Retrain a model, eg on TRAINVAL instead of on TRAIN, using early-stopping values recorded from eg cross-validation.
             *
             * @param m the model to be learned further.
             * @param cfg how to train (probably the same thing given to fit() previously)
             * @param result the result of fit(), or the best result of crossvalidation_fit, with per-stage results
             */
            virtual msg::FitResult learn_until_previous_loss_reached(model& m, const msg::Fit& cfg, const msg::FitResult& result);

            /**
             * Overload this to switch to a different mode on the dataset.
             *
             * Does nothing by default.
             *
             * @param mode determines whether we're in training, validation or testing
             * @param split if less than zero, the split should not be changed. 
             */
            virtual void switch_dataset(cv_mode mode, int split=-1);

            /**
             * This function calls switch_dataset, and is overloaded by multistage
             * learner, and you should likely neither overload nor touch it.
             *
             * @param mode determines whether we're in training, validation or testing
             * @param split if less than zero, the split should not be changed. 
             */
            virtual void _switch_dataset(cv_mode mode, int split=0);

            /**
             *
             * This function calls load_batch, and is overloaded by multistage
             * learner, and you should likely neither overload nor touch it.
             *
             */
            virtual void _load_batch(model* m, unsigned int epoch, unsigned int batchid);

            /**
             * Return the current gradient_descent object, eg during fit().
             */
            inline boost::shared_ptr<gradient_descent> gd()const{ return m_gd; }

            /**
             * Return the current monitor object, eg during fit().
             */
            inline boost::shared_ptr<monitor> mon()const{ return m_mon; }

            /**
             * (virtual) dtor.
             */
            virtual ~learner2();

            void register_validation_batchsize(model& m, gradient_descent& gd, early_stopper& es,
                    const msg::GradientDescent& cfg, const msg::EarlyStopper& escfg);

            inline 
                boost::shared_ptr<early_stopper> get_early_stopper(){ return m_es; }

    };

    /**
     * Cross-validation that works together with the learner2 and model class.
     */
    struct crossvalidator2{
        private:
        public:
            typedef models::model model;
            /**
             * Call fit() for a number of splits, saving the best result, and returning the performance on all folds.
             *
             * @param lrn the learner, on which we'll call fit()
             * @param m the model to be fitted
             * @param cfg how to fit the model
             * @return one result for every split
             */
            msg::XValResult fit(learner2& lrn, boost::shared_ptr<model> m, const msg::XVal& cfg);
    };

    struct multistage_dataset{
        std::string m_path;
        std::string m_dataset;
        std::string m_stagename;
        std::vector<std::vector<host_matrix> > m_data;
        std::vector<boost::shared_ptr<ParameterInput> > m_inputs;
        multistage_dataset(const std::string& path, const std::string& dataset,
                const std::string& stage,
                std::vector<boost::shared_ptr<ParameterInput> > inputs);

        multistage_dataset(
                const std::vector<std::vector<host_matrix> >& traindata,
                const std::vector<std::vector<host_matrix> >& valdata
                );
        void load_batch(unsigned int epoch, unsigned int batch);
        unsigned int n_batches()const;
    };

    /**
     * A multi-stage learner can for example learn a model that requires
     * 'pre-training'. There should be a sequence of stages which need to be
     * trained, each one has a loss function to be optimized and its own
     * optimization settings. Optionally, a (lower) stage may be substituted by
     * its outputs, resulting in a smaller model and a new dataset.
     */
    class multistage_learner
        : public learner2
    {
        protected:
            std::vector<
                boost::shared_ptr<multistage_dataset> >
                m_stage_datasets;
            boost::shared_ptr<multistage_dataset> m_current_dataset;
            cv_mode m_current_cvmode;
        public:
            typedef models::multistage_model multistage_model;

            /**
             * ctor.
             */
            multistage_learner():m_stage_datasets(4), m_current_cvmode(CM_TRAIN){}

            /**
             * @overload
             */
            msg::FitResult fit(model& m, const msg::Fit& cfg);

            /**
             * Switch to a new learning stage.
             *
             * 1. record all "outputs" of the model to a file
             * 2. switch to the new dataset
             * 3. substitute all "outputs" with inputs for the next stage
             */
            std::vector<boost::shared_ptr<graph_modifiers::substitute_op_with_input> >
                switch_stage_with_outputs(multistage_model& m,
                        const multistage_model::stage_type& current_stage,
                        const std::string& path, int batch_size);

            /**
             * this _load_batch version only calls load_batch if in the 1st stage, 
             * but uses our own functions for the other stages.
             */
            virtual void _load_batch(model* m, unsigned int epoch, unsigned int bid);

            /**
             * this _switch_dataset version only calls switch_dataset if in
             * the 1st stage, but uses our own functions for the other stages.
             */
            virtual void _switch_dataset(cv_mode mode, int split=0);

            /**
             * this _n_batches version only calls n_batches if in the 
             * 1st stage, but uses our own function for the other stages.
             *
             * @param batchsize the number of instances in one batch
             * @return 1 by default
             */
            virtual unsigned int _n_batches(unsigned int batchsize);

            /**
             * Retrain a model, eg on TRAINVAL instead of on TRAIN, using early-stopping values recorded from eg cross-validation.
             *
             * @param m the model to be learned further.
             * @param cfg how to train (probably the same thing given to fit() previously)
             * @param result the result of fit(), or the best result of crossvalidation_fit, with per-stage results
             */
            msg::FitResult learn_until_previous_loss_reached(model& m, const msg::Fit& cfg, const msg::FitResult& result);

        private:
            /**
             * returns the current cross-validation mode for internal purposes.
             */
            inline cv_mode current_cvmode()const{return m_current_cvmode;}
    };
}

#endif /* __LEARNER2_HPP__ */
