#ifndef __LEARNER2_HPP__
#define __LEARNER2_HPP__

#include <boost/signals.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cuvnet/models/models.hpp>
#include <datasets/dataset.hpp>    /* for cv_mode */

namespace cuvnet
{
    class monitor;
    class gradient_descent;
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
            boost::signals::scoped_connection con;
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
            boost::signals::scoped_connection con;
            exponential_learnrate_schedule(gradient_descent* _gd, float begin, float end, int epochs, float t0);
            virtual void operator()(unsigned int epoch, unsigned int wups);
        };

        /**
         * A linearly increasing momentum schedule.
         * @ingroup learning
         */
        struct linear_momentum_schedule : public hyperparam_schedule{
            float initial, final, duration;
            momentum_gradient_descent* gd;
            boost::signals::scoped_connection con;
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
        boost::signals::scoped_connection con0, con1;
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
        boost::signals::scoped_connection con;
        stop_when_target_loss_reached(gradient_descent& gd, monitor& _mon, float tloss);
        void operator()(unsigned int current_epoch, unsigned int wups);
    };

    /**
     * sets up common learning features and fits given models.
     * @ingroup learning
     */
    struct learner2{
        protected:
            typedef models::model model;
            typedef boost::property_tree::ptree ptree;

            /**
             * Overload this to load a specific batch from your dataset into the model.
             *
             * Does nothing by default.
             *
             * @param m the model to load the batch into
             * @param bid the batch id which is to be loaded into the model
             */
            virtual void load_batch(model* m, unsigned int bid);

            /**
             * Overload this to tell learner how many batches your dataset has.
             *
             * @param batchsize the number of instances in one batch
             * @return 1 by default
             */
            virtual unsigned int n_batches(unsigned int batchsize);

            /**
             * Overload this to switch to a different mode on the dataset.
             *
             * Does nothing by default.
             */
            virtual void switch_dataset(cv_mode mode, int split=0, const std::string& stage = "");

            /**
             * In this hook you can modify the gradient_descent object just before learning.
             * 
             * Does nothing by default.
             */
            virtual void before_learning(model* m, gradient_descent& gd, cuvnet::early_stopper* es);

            /**
             * Returns a gradient Gradient Descent object described by the configuration parameter.
             *
             * Not all gradient_descent objects in cuvnet are currently
             * supported, but it should be easy to add them, either by
             * modifying this function or by overloading it.
             *
             * @param m the model for which to generate the gd object
             * @param cfg the configuration subtree describing how the gd object should be created
             * @param stage the current training stage (e.g. pretraining) to be passed on to the model
             */
            virtual
            boost::shared_ptr<gradient_descent> 
                get_gradient_descent(model& m, const ptree& cfg, const std::string& stage = "");

            /**
             * Returns an early stopper or NULL, configured according to the cfg parameter.
             */
            boost::shared_ptr<early_stopper>
                get_early_stopper(gradient_descent& gd, monitor& mon, const ptree& cfg, const std::string& stage = "");

            /**
             * Returns a monitor that watches loss and error of the model.
             *
             * It also asks the model to register possible other watchpoints
             * with the monitor using model::register_watches().
             *
             * @param m the model that we want to monitor
             * @param cfg the configuration subtree of the monitor
             * @param stage the stage of training we're in (e.g. pretraining),
             *              passed to model
             */
            boost::shared_ptr<monitor> 
                get_monitor(model& m, const ptree& cfg, const std::string& stage = "");

            /**
             * Returns a learnrate schedule, which will be called in the before_epoch event.
             */
            boost::shared_ptr<schedules::hyperparam_schedule> 
                virtual get_learnrate_schedule(gradient_descent& gd, int max_epochs, ptree cfg);

            /**
             * Returns a momentum schedule, which will be called in the before_epoch event.
             */
            boost::shared_ptr<schedules::hyperparam_schedule> 
                virtual get_momentum_schedule(gradient_descent& pgd, int max_epochs, ptree cfg);

        public:

            /**
             * Determine how many splits there are for crossvalidation.
             *
             * @return 1 by default
             */
            virtual unsigned int n_splits()const;

            /**
             * Serialize the model to a file. 
             * This is needed by crossvalidation to save the best intermediate model.
             */
            virtual void save_model(model& m, std::string filename);

            /**
             * Deserialize the model from a file. 
             * This is needed by crossvalidation to load the best intermediate model.
             */
            virtual void load_model(model& m, std::string filename);

            /**
             * Fit a model, using features described in configuration param.
             *
             * @param m the model to be fitted
             * @param cfg parameters for fitting
             * @param stage an identifier for the current stage of training (e.g. for pretraining)
             */
            ptree fit(model& m, const ptree& cfg, const std::string& stage = "");

            /**
             * Call fit() for a number of splits, saving the best result, and returning the performance on all folds.
             *
             * @param m the model to be fitted
             * @param cfg how to fit the model
             * @return one result for every split
             */
            ptree crossvalidation_fit(model& m, const ptree& cfg);

            /**
             * Continue learning in an already learned model, eg on TRAINVAL instead of on TRAIN.
             *
             * @param m the model to be learned further.
             * @param cfg how to train (probably the same thing given to fit() previously)
             * @param result the result of fit(), or the best result of crossvalidation_fit
             */
            ptree continue_learning_until_previous_loss_reached(model& m, const ptree& cfg, const ptree& result, const std::string& stage="");

            /**
             * (virtual) dtor.
             */
            virtual ~learner2();
    };
}

#endif /* __LEARNER2_HPP__ */
