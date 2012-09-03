#include <mongo/client/dbclient.h>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>                                                                  
#include <boost/lambda/lambda.hpp>                                                                  
#include <mdbq/client.hpp>

#include <cuvnet/ops.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/contractive_auto_encoder.hpp>
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
            std::vector<float> m_aes_wd; ///< auto encoder weight decay
            unsigned int m_aes_bs; ///< batch size during AE learning
            unsigned int m_mlp_bs; ///< batch size during MLP learning
            float m_mlp_lr; ///< learning rate of regression
            float m_mlp_wd; ///< weight decay of regression
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
            virtual bool refit_for_test()const { return false; };
    
            /// reset parameters
            virtual void reset_params();
    
            /// returns performance of current parameters
            float perf(monitor* mon=NULL);

    private:
        void load_batch_supervised(unsigned int batch){
            boost::dynamic_pointer_cast<ParameterInput>(m_aes.input())->data() = m_sdl.get_data_batch(batch).copy();
            boost::dynamic_pointer_cast<ParameterInput>(m_regression.get_target())->data() = m_sdl.get_label_batch(batch).copy();
        }
        void load_batch_unsupervised(unsigned int batch){
            boost::dynamic_pointer_cast<ParameterInput>(m_aes.input())->data() = m_sdl.get_data_batch(batch).copy();
        }
        void set_batchsize(unsigned int b){
            m_sdl.set_batchsize(b);
            matrix& m = boost::dynamic_pointer_cast<ParameterInput>(m_aes.input())->data();
            m.resize(cuv::extents[b][m.shape(1)]);
            matrix& t = boost::dynamic_pointer_cast<ParameterInput>(m_regression.get_target())->data();
            t.resize(cuv::extents[b][t.shape(1)]);
        }
    };

    namespace detail{
        struct value_not_found_exception{};

        void check(const mongo::BSONObj& o, const std::string& name){
            if(!o.hasField("vals"))
                throw value_not_found_exception();
            if(!o["vals"].isABSONObj())
                throw value_not_found_exception();
            if(!o["vals"].Obj().hasField(name.c_str()))
                throw value_not_found_exception();
            if(o["vals"][name].type() != mongo::Array)
                throw value_not_found_exception();
            if(o["vals"][name].Obj().nFields() < 1)
                throw value_not_found_exception();
        }

        template<class T>
            T get(const mongo::BSONObj& o, const std::string& name){ }

        template<>
            float get<float>(const mongo::BSONObj& o, const std::string& name){ 
                check(o, name);
                return o["vals"][name].Array()[0].Double();
            }
        template<>
            int get<int>(const mongo::BSONObj& o, const std::string& name){ 
                check(o, name);
                return o["vals"][name].Array()[0].Int();
            }
    }
    template<class T>
        T get(const mongo::BSONObj& o, const std::string& name, int idx=-1){
            if(idx>=0)
                return detail::get<T>(o, name + "_" + boost::lexical_cast<std::string>(idx));
            return detail::get<T>(o, name);
        }
    
    unsigned int idx_to_bs(unsigned int idx){
        switch(idx){
            case 0: return 1;
            case 1: return 2;
            case 2: return 4;
            case 3: return 16;
            case 4: return 64;
        }
        throw std::runtime_error("Unknown batchsize index!");
    }


    template<class R>
    void
    pretrained_mlp_learner<R>::constructFromBSON(const mongo::BSONObj& o){
        m_aes_bs = idx_to_bs(get<int>(o, "aes_bs"));
        m_mlp_bs = idx_to_bs(get<int>(o, "mlp_bs"));

        m_sdl.init(m_aes_bs, "mnist_rot", 1);

        m_mlp_lr = get<float>(o, "mlp_lr");
        //m_mlp_wd = get<float>(o, "mlp_wd");
        int n_layers;
        for (n_layers = 0; n_layers < 100; ++n_layers){
            try{
                m_aes_lr.push_back(get<float>(o,"aes_lr", n_layers));
                m_aes_wd.push_back(get<float>(o,"aes_wd", n_layers));
                int layer_size0 = get<float>(o,"aes_ls0", n_layers);
                int layer_size1 = get<float>(o,"aes_ls1", n_layers);
                m_aes.add<two_layer_contractive_auto_encoder>(layer_size0, layer_size1, m_sdl.get_ds().binary, m_aes_wd.back());
            }catch(const detail::value_not_found_exception& e){
                break;
            }
        }
        std::cout << "n_layers:" << n_layers << std::endl;

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
        // abuse gradient descent for looping over the dataset in batches.
        gradient_descent gd(m_regression.classification_error_direct(), 0, params, 0.0f); // learning rate 0
        gd.before_batch.connect(boost::bind(&pretrained_mlp_learner::load_batch_supervised,this,_2));
        gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches, &m_sdl));
        if(mon) {
            gd.register_monitor(*mon);
            gd.minibatch_learning(1, INT_MAX);
            return mon->mean("classification error");
        }else{
            monitor mon;
            mon.add(monitor::WP_SCALAR_EPOCH_STATS, gd.loss(), "classification error");
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

        set_batchsize(m_aes_bs);
        for (unsigned int ae_id = 0; ae_id < n_ae; ++ae_id)
        {
    
            // set the initial number of epochs to zero
            generic_auto_encoder& ae = m_aes.get_ae(ae_id);
    
            // set up gradient descent
            gradient_descent gd(ae.loss(), 0, ae.unsupervised_params(), m_aes_lr[ae_id], 0.f);
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
                gd.setup_convergence_stopping(boost::bind(&monitor::mean, &mon, "total loss"), 0.99f, 6, 2.0);
                gd.minibatch_learning(100, INT_MAX);
                m_epochs[ae_id] += gd.iters();
            }
        }
    
        
        {
            // Supervised finetuning
            m_aes.deinit();
            set_batchsize(m_mlp_bs);
            std::vector<Op*> aes_params = m_aes.supervised_params();
            std::vector<Op*> params     = m_regression.params();
            std::copy(aes_params.begin(), aes_params.end(), std::back_inserter(params));
            
            gradient_descent gd(m_regression.get_loss(),0,params,m_mlp_lr, -0);
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
                gd.minibatch_learning(10000, INT_MAX); 
                m_epochs.back() += gd.iters();
            }
        }
    }
}

struct hyperopt_client                                                                       
: public mdbq::Client{                                                                             

    int m_dev;
    bool m_init;
    hyperopt_client(int dev)
        : mdbq::Client("131.220.7.92", "hyperopt")
        , m_dev(dev)
        , m_init(false)
    { }

    void handle_task(const mongo::BSONObj& o){                                               
        if(!m_init && cuv::IsSame<cuvnet::matrix::memory_space_type,cuv::dev_memory_space>::Result::value){
            cuv::initCUDA(m_dev);
            cuv::initialize_mersenne_twister_seeds(time(NULL));
            m_init = true;
        }

        auto ml = boost::make_shared<cuvnet::pretrained_mlp_learner<cuvnet::logistic_regression> >(true);
        ml->constructFromBSON(o);

        cuvnet::cv::all_splits_evaluator ase(ml);
        float perf = ase();

        finish(BSON(  "status"  << "ok"
                    <<"loss"    << perf
                    <<"dev"     << m_dev));
        exit(0);
    }

};

int
main(int argc, char **argv)
{
    if(std::string("worker") == argv[1]){
        cuvAssert(argc==3);

        boost::asio::io_service ios; 
        hyperopt_client hc(boost::lexical_cast<int>(argv[2]));
        hc.reg(ios,1);
        ios.run();

    }

    if(std::string("test") == argv[1]){
        cuvAssert(argc==3);
        if(cuv::IsSame<cuvnet::matrix::memory_space_type,cuv::dev_memory_space>::Result::value){
            cuv::initCUDA(boost::lexical_cast<int>(argv[2]));
            cuv::initialize_mersenne_twister_seeds(time(NULL));
        }

        mongo::BSONObjBuilder bob;
        bob<<"aes_bs" <<BSON_ARRAY(3);
        bob<<"mlp_bs" <<BSON_ARRAY(3);
        bob<<"mlp_lr" <<BSON_ARRAY(0.01f);
        bob<<"mlp_wd" <<BSON_ARRAY(0.0001f);

        bob<<"aes_lr_0" <<BSON_ARRAY(0.01f);
        bob<<"aes_wd_0" <<BSON_ARRAY(0.01f);
        bob<<"aes_ls_0" <<BSON_ARRAY( 64.f);

        bob<<"aes_lr_1" <<BSON_ARRAY(0.01f);
        bob<<"aes_wd_1" <<BSON_ARRAY(0.01f);
        bob<<"aes_ls_1" <<BSON_ARRAY( 92.f);


        auto ml = boost::make_shared<cuvnet::pretrained_mlp_learner<cuvnet::logistic_regression> >(true);
        ml->constructFromBSON(BSON("vals"<<bob.obj()));

        cuvnet::cv::all_splits_evaluator ase(ml);
        ase();
    }

    return 0;
}
