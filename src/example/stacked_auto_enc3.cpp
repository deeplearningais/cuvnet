#include <mongo/client/dbclient.h>
//#include <boost/asio.hpp>
//#include <boost/bind.hpp>
//#include <boost/thread.hpp>                                                                  
#include <boost/lambda/lambda.hpp>                                                                  
#include <mdbq/client.hpp>

#include <cuvnet/ops.hpp>
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/crossvalid.hpp>
#include <tools/learner.hpp>
#include <tools/monitor.hpp>


namespace cuvnet{
    class simple_crossvalidatable_learner
    : public crossvalidatable{
        protected:
            typedef SimpleDatasetLearner<matrix::memory_space_type> sdl_t; ///< provides access to dataset
            sdl_t m_sdl; ///< provides access to dataset
    
        private:
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) { 
                    ar & boost::serialization::base_object<crossvalidatable>(*this);
                }
        public:
    
            /// switch  to a different split/crossvalidation mode
            virtual void switch_dataset(unsigned int split, cv_mode mode);
    
            /// return the number of requested splits
            virtual unsigned int n_splits();
    };
    
    void
    simple_crossvalidatable_learner::switch_dataset(unsigned int split, cv_mode mode){
        m_sdl.switch_dataset(split,mode);
    }
    unsigned int
    simple_crossvalidatable_learner::n_splits(){
        return m_sdl.n_splits();
    }
    
    
    template<class RegressionType=logistic_regression>
    class pretrained_mlp_learner
    : public simple_crossvalidatable_learner{
        private:
            friend class boost::serialization::access;
            std::vector<int>   m_epochs; ///< number of epochs learned
            std::vector<float> m_aes_lr; ///< auto encoder learning rates
            float m_mlp_lr; ///< learning rate of regression
            auto_encoder_stack m_aes; ///< contains all auto-encoders
            RegressionType m_regression; ///< does the regression for us
    
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) { 
                    ar & boost::serialization::base_object<simple_crossvalidatable_learner>(*this);
                }
        public:
            /**
             * constructor.
             * @param binary if true, assume inputs are bernoulli-distributed.
             */
            pretrained_mlp_learner(bool binary)
                :m_aes(binary)
            {}

            /// return the AE stack for initialization etc.
            inline auto_encoder_stack& aes(){ return m_aes; }
    
            /// initialize the object from a BSON object description
            virtual void constructFromBSON(const mongo::BSONObj&);
    
            /// learn on current split
            virtual void fit();
    
            /// predict on current test set
            virtual float predict();
    
            /// return minimum on VAL before retraining for TEST is performed
            virtual float refit_thresh()const{ return INT_MAX; };
    
            /// should return true if retrain for fit is required
            /// @return true
            virtual bool refit_for_test()const { return true; };
    
            /// reset parameters
            virtual void reset_params();
    
            /// returns performance of current parameters
            float perf(monitor* mon=NULL);

    private:
        void load_batch_supervised(unsigned int batch){
            boost::dynamic_pointer_cast<ParameterInput>(m_aes.input())->data() = m_sdl.get_data_batch(batch);
            boost::dynamic_pointer_cast<ParameterInput>(m_regression.get_target())->data() = m_sdl.get_label_batch(batch);
        }
        void load_batch_unsupervised(unsigned int batch){
            boost::dynamic_pointer_cast<ParameterInput>(m_aes.input())->data() = m_sdl.get_data_batch(batch);
        }
    };
    
    template<class R>
    void
    pretrained_mlp_learner<R>::constructFromBSON(const mongo::BSONObj& o){
        m_sdl.constructFromBSON(o);
        int n_layers = 2;
        m_mlp_lr = o["mlp_lr"].Double();
        m_sdl.set_early_stopping_frac(o["es_frac"].Double());
        for (int l = 0; l < n_layers; ++l)
        {
            m_aes_lr.push_back(o["aes_lr"].Double());
            m_aes.add<simple_auto_encoder>(512, m_sdl.get_ds().binary);
        }

        // create a ParameterInput for the input
        cuvAssert(m_sdl.get_ds().train_data.ndim() == 2);
        boost::shared_ptr<ParameterInput> input
            = boost::make_shared<ParameterInput>(
                    cuv::extents[m_sdl.batchsize()][m_sdl.get_ds().train_data.shape(1)]);

        // create a ParameterInput for the target
        cuvAssert(m_sdl.get_ds().train_labels.ndim() == 2);
        boost::shared_ptr<ParameterInput> target 
            = boost::make_shared<ParameterInput>(
                    cuv::extents[m_sdl.batchsize()][m_sdl.get_ds().train_labels.shape(1)]);

        m_aes.init(input);
        m_regression.init(m_aes.get_encoded(), target);
        
        // set the initial number of epochs to zero
        m_epochs.resize(n_layers+1);
        for (int i = 0; i < n_layers+1; ++i)
            m_epochs[i] = 0;
    }

    template<class R>
    void
    pretrained_mlp_learner<R>::reset_params(){
        m_aes.reset_weights();
        m_regression.reset_weights();
    }
    template<class R>
    float
    pretrained_mlp_learner<R>::predict(){ return perf(); }

    template<class R>
    float 
    pretrained_mlp_learner<R>::perf(monitor* mon){
        std::vector<Op*> params; // empty!
        gradient_descent gd(m_regression.get_loss(), 0, params, 0.0f); // learning rate 0
        gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches, &m_sdl));
        std::cout << m_sdl.describe_current_mode_split(true) << std::endl;
        if(mon) {
            gd.register_monitor(*mon);
            gd.minibatch_learning(1, INT_MAX);
            return mon->mean("classification error");
        }else{
            monitor mon;
            mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, m_regression.classification_error(), "classification error");
            gd.register_monitor(mon);
            gd.minibatch_learning(1, INT_MAX);
            return mon.mean("classification error");
        }
    }

    template<class R>
    void pretrained_mlp_learner<R>::fit(){
        using namespace boost::assign;
        unsigned int n_ae = m_aes.size();
        bool in_trainall = m_sdl.get_current_cv_mode() == CM_TRAINALL;
        std::cout << m_sdl.describe_current_mode_split(true) << std::endl;

        for (unsigned int ae_id = 0; ae_id < n_ae; ++ae_id)
        {
    
            // set the initial number of epochs to zero
            generic_auto_encoder& ae = m_aes.get_ae(ae_id);
    
            // set up gradient descent
            gradient_descent gd(ae.loss(), 0, ae.unsupervised_params(), m_aes_lr[ae_id], 0.0f);
            gd.before_batch.connect(boost::bind(&pretrained_mlp_learner::load_batch_unsupervised,this,_2));
            gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));
    
            // set up monitor
            monitor mon(false);
            mon.set_training_phase(m_sdl.get_current_cv_mode(), m_sdl.get_current_split());
            mon.add("layer", ae_id);
            mon.add(monitor::WP_SCALAR_EPOCH_STATS, ae.loss(), "total loss");
            gd.register_monitor(mon);

            // do the actual learning
            if(in_trainall) {
                int n = m_epochs[ae_id] / m_sdl.n_splits();
                gd.minibatch_learning(n, INT_MAX);
            }
            else {
                gd.setup_convergence_stopping(boost::bind(&monitor::mean, &mon, "total loss"), 0.95f, 6, 2.0);
                gd.minibatch_learning(2, INT_MAX);
                m_epochs[ae_id] += gd.iters();
            }
        }
    
        
        {
            // Supervised finetuning
            std::vector<Op*> aes_params = m_aes.supervised_params();
            std::vector<Op*> params     = m_regression.params();
            std::copy(aes_params.begin(), aes_params.end(), std::back_inserter(params));
            
            gradient_descent gd(m_regression.get_loss(),0,params,m_mlp_lr,0.00000f);
            gd.before_batch.connect(boost::bind(&pretrained_mlp_learner::load_batch_supervised,this,_2));
            gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));
    
            // set up monitor
            monitor mon(false);
            mon.add("layer", n_ae);
            mon.set_training_phase(m_sdl.get_current_cv_mode(), m_sdl.get_current_split());
            mon.add(monitor::WP_SCALAR_EPOCH_STATS, m_regression.get_loss(), "total loss");
            mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, m_regression.classification_error(), "classification error");
            gd.register_monitor(mon);
    
            if(m_sdl.can_earlystop()){
                gd.setup_early_stopping(boost::bind(&monitor::mean, &mon, "classification error"), 10, 0.995f, 2.f);
    
                gd.before_early_stopping_epoch.connect(boost::bind(&sdl_t::before_early_stopping_epoch,&m_sdl));
                gd.before_early_stopping_epoch.connect(boost::bind(&monitor::set_training_phase, &mon, CM_VALID, m_sdl.get_current_split()));
    
                gd.after_early_stopping_epoch.connect(0,boost::bind(&sdl_t::after_early_stopping_epoch,&m_sdl));
                gd.after_early_stopping_epoch.connect(1,boost::bind(&monitor::set_training_phase,&mon, CM_TRAIN, m_sdl.get_current_split()));
            }
            // do the actual learning
            if(in_trainall)
            {
                int n = m_epochs.back() / m_sdl.n_splits();
                gd.minibatch_learning(n, INT_MAX);
            }
            else {
                gd.minibatch_learning(100, INT_MAX);
                m_epochs.back() += gd.iters();
            }
            m_sdl.set_early_stopping_frac(0.f);
        }
    
    
    }
}


int
main(int argc, char **argv)
{




    if(std::string("test") == argv[1]){
        cuvAssert(argc==3);
        if(cuv::IsSame<cuvnet::matrix::memory_space_type,cuv::dev_memory_space>::Result::value){
            cuv::initCUDA(boost::lexical_cast<int>(argv[2]));
            cuv::initialize_mersenne_twister_seeds(time(NULL));
        }

        mongo::BSONObjBuilder bob;
        bob<<"nsplits"<<5;
        bob<<"dataset"<<"mnist";
        bob<<"bs"     <<128;

        bob<<"nlayers"<<2;
        bob<<"aes_lr" <<0.01f;
        bob<<"mlp_lr" <<0.01f;
        bob<<"es_frac"<<.1f;

        auto ml = boost::make_shared<cuvnet::pretrained_mlp_learner<cuvnet::logistic_regression> >(true);
        ml->constructFromBSON(bob.obj());

        cuvnet::cv::all_splits_evaluator ase(ml);
        ase();
    }

    return 0;
}
