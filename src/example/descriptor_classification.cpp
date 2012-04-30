// vim:ts=4:sw=4:et
#include <signal.h>
#include <fstream>
#include <cmath>
#include <boost/assign.hpp>
#include <boost/bind.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/export.hpp>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <tools/visualization.hpp>
#include <tools/gradient_descent.hpp>
#include <mongo/bson/bson.h>

#include <tools/crossvalid.hpp>
#include <tools/learner.hpp>

#include "stacked_auto_enc2.hpp"
#include "pretrained_mlp.hpp"


using namespace boost::assign;



class pretrained_mlp_trainer
: public crossvalidatable
{
    private:
        boost::shared_ptr<pretrained_mlp> m_mlp; ///< the mlp to be trained
        boost::shared_ptr<auto_enc_stack> m_aes; ///< the stacked ae to be pre-trained
        SimpleDatasetLearner<matrix::memory_space_type> m_sdl; /// provides access to dataset

        std::vector<float> m_aes_lr; /// learning rates of stacked AE
        float              m_mlp_lr; /// learning rates of stacked MLP
        bool               m_pretraining; /// whether pretraining is requested
        bool               m_finetune; /// whether finetuning is requested
        bool               m_unsupervised_finetune; /// whether finetuning is requested

        typedef SimpleDatasetLearner<matrix::memory_space_type> sdl_t;
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ar & boost::serialization::base_object<crossvalidatable>(*this);
            }
    public:
        void constructFromBSON(const mongo::BSONObj& o)
        {
            // construct all members

            std::cout <<"---------------------------------"<<std::endl;
            std::cout <<"Working on: "<<o<<std::endl;
            m_sdl.constructFromBSON(o);

            unsigned int bs = m_sdl.batchsize();
            unsigned int dd = m_sdl.datadim();

            std::vector<mongo::BSONElement> ar = o["stack"].Array();
            std::vector<int>   layer_sizes(ar.size());
            std::vector<float> noise(ar.size());
            std::vector<float> lambda(ar.size());
            std::vector<bool> twolayer_ae(ar.size());
            m_aes_lr.resize(ar.size());

            for(unsigned int i=0;i<ar.size(); i++){
                layer_sizes[i] = ar[i].Obj()["size"].Int();
                noise[i]       = ar[i].Obj()["noise"].Double();
                lambda[i]      = ar[i].Obj()["lambda"].Double();
                m_aes_lr[i]    = ar[i].Obj()["lr"].Double();
                twolayer_ae[i]  = ar[i].Obj()["twolayer"].Bool();
            }
            bool binary = m_sdl.get_ds().binary;
            m_aes.reset(
                new auto_enc_stack(bs,dd,ar.size(),&layer_sizes[0], binary, &noise[0], &lambda[0], twolayer_ae));
            m_mlp.reset(
                new pretrained_mlp(m_aes->encoded(),21, true)); // TODO: fixed number of 10 classes!!???
            m_mlp_lr = o["mlp_lr"].Double();
            m_pretraining = o["pretrain"].Bool();
            m_finetune    = o["sfinetune"].Bool();
            m_unsupervised_finetune    = o["ufinetune"].Bool();
        }


        /**
         * returns classification error on current dataset
         */
        float predict() {
            // "learning" with learnrate 0 and no weight updates
            if(m_finetune){
                std::vector<Op*> params;
                gradient_descent gd(m_mlp->output(),0,params,0.0f,0.0f);
                gd.before_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
                gd.after_epoch.connect(boost::bind(&pretrained_mlp::log_loss, m_mlp.get(),_1));
                gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_supervised,this,_2));
                gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
                gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));
                gd.minibatch_learning(1,0,0); // 1 epoch
                return m_mlp->perf();
            }else /*if(m_unsupervised_finetune)*/{
                std::vector<Op*> params;
                gradient_descent gd(m_aes->combined_rec_loss(),0,params, 0.0f,0.00000f);
                gd.before_epoch.connect(boost::bind(&auto_enc_stack::reset_loss, m_aes.get()));
                gd.after_epoch.connect(boost::bind(&auto_enc_stack::log_loss, m_aes.get(), _1));
                gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_unsupervised,this,_2));
                gd.after_batch.connect(boost::bind(&auto_enc_stack::acc_loss,m_aes.get()));
                gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));
                gd.minibatch_learning(1,0,0); // 1 epoch
                return m_aes->perf();
            }/*else{*/
                //cuvAssert(false);
            /*}*/
            return -2.f;
        }
        void param_logging(std::string desc, std::vector<Op*> ops){
            for(unsigned int i=0;i<ops.size(); i++)
                param_logging(desc, ops[i]);
        }
        void param_logging(std::string desc, Op* op){
            Input* inp = (Input*)op;
            cuv::tensor<float,cuv::host_memory_space> cpy = inp->data();
            mongo::BSONObjBuilder bob;
            bob.append("param",inp->name());
            bob.append("desc",desc);
            mongo::BSONArrayBuilder bab;
            for(int i=0;i<cpy.ndim();i++)
                bab.append(cpy.shape(i));
            bob.appendArray("shape",bab.arr());
            //bob.appendBinData("data", cpy.size()*sizeof(float), mongo::BinDataGeneral, (const char*)cpy.ptr());
            g_worker->log((char*)cpy.ptr(), cpy.size()*sizeof(float), bob.obj());
        }
        /**
         * train the given auto_encoder stack and the mlp
         */
        void fit() {
            ////////////////////////////////////////////////////////////
            //             un-supervised pre-training
            ////////////////////////////////////////////////////////////
            if(m_pretraining) {
                for(unsigned int l=0; l<m_aes->size(); l++) {
                    std::cout <<".pretraining layer "<<l<<std::endl;
                    g_worker->log(BSON("who"<<"trainer"<<"topic"<<"layer_change"<<"layer"<<l));
                    std::vector<Op*> params = m_aes->get(l).unsupervised_params();

                    gradient_descent gd(m_aes->get(l).loss(),0,params,m_aes_lr[l],0.00000f);
                    gd.before_epoch.connect(boost::bind(&auto_encoder::reset_loss, &m_aes->get(l)));
                    gd.after_epoch.connect(boost::bind(&auto_encoder::log_loss, &m_aes->get(l), _1));
                    gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_unsupervised,this,_2));
                    gd.after_batch.connect(boost::bind(&auto_encoder::acc_loss,&m_aes->get(l)));
                    gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));

                    if(m_sdl.can_earlystop()) {
                        // we can only use early stopping when validation data is given
                        //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
                        gd.setup_early_stopping(boost::bind(&auto_encoder::perf,&m_aes->get(l)), 5, 0.0001f, 3);
                        gd.before_validation_epoch.connect(boost::bind(&auto_encoder::reset_loss, &m_aes->get(l)));
                        gd.before_validation_epoch.connect(boost::bind(&sdl_t::before_validation_epoch,&m_sdl));
                        gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp_trainer::validation_epoch,this,true));
                        gd.after_validation_epoch.connect(1, boost::bind(&sdl_t::after_validation_epoch, &m_sdl));
                        gd.after_validation_epoch.connect(1, boost::bind(&pretrained_mlp_trainer::validation_epoch,this,false));
                        gd.after_validation_epoch.connect(0, boost::bind(&auto_encoder::log_loss, &m_aes->get(l), _1));
                        gd.minibatch_learning(10000);
                        m_aes->get(l).s_epochs(gd.rounds()); // remember number of iterations until optimum
                    } else {
                        std::cout << "TRAINALL phase: aes"<<l<<" avg_epochs="<<m_aes->get(l).avg_epochs()<<std::endl;
                        gd.minibatch_learning(m_aes->get(l).avg_epochs()); // TRAINALL phase. Use as many as in previous runs
                    }
                    param_logging("after_pretrain", params);
                }
            }
            ////////////////////////////////////////////////////////////
            //               unsupervised finetuning
            ////////////////////////////////////////////////////////////
            if(m_unsupervised_finetune){
                std::cout <<".unsupervised finetuning"<<std::endl;
                std::vector<Op*> params;
                for(unsigned int l=0; l<m_aes->size(); l++) // derive w.r.t. /all/ parameters except output bias of AEs
                {
                    std::vector<Op*> tmp = m_aes->get(l).unsupervised_params();
                    std::copy(tmp.begin(), tmp.end(), std::back_inserter(params));
                }

                gradient_descent gd(m_aes->combined_rec_loss(),0,params, m_mlp_lr,0.00000f);
                gd.before_epoch.connect(boost::bind(&auto_enc_stack::reset_loss, m_aes.get()));
                gd.after_epoch.connect(boost::bind(&auto_enc_stack::log_loss, m_aes.get(), _1));
                gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_unsupervised,this,_2));
                gd.after_batch.connect(boost::bind(&auto_enc_stack::acc_loss,m_aes.get()));
                gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));

                if(m_sdl.can_earlystop()) {
                    // we can only use early stopping when validation data is given
                    //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
                    gd.setup_early_stopping(boost::bind(&auto_enc_stack::perf,m_aes.get()), 5, 0.0001f, 3);
                    gd.before_validation_epoch.connect(boost::bind(&auto_enc_stack::reset_loss, m_aes.get()));
                    gd.before_validation_epoch.connect(boost::bind(&sdl_t::before_validation_epoch,&m_sdl));
                    gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp_trainer::validation_epoch,this,true));
                    gd.after_validation_epoch.connect(1, boost::bind(&sdl_t::after_validation_epoch, &m_sdl));
                    gd.after_validation_epoch.connect(1, boost::bind(&pretrained_mlp_trainer::validation_epoch,this,false));
                    gd.after_validation_epoch.connect(0, boost::bind(&auto_enc_stack::log_loss, m_aes.get(), _1));
                    gd.minibatch_learning(10000);
                    m_aes->s_epochs(gd.rounds()); // remember number of iterations until optimum
                } else {
                    std::cout << "TRAINALL phase: aes unsupervised finetuning; avg_epochs="<<m_aes->avg_epochs()<<std::endl;
                    gd.minibatch_learning(m_aes->avg_epochs()); // TRAINALL phase. Use as many as in previous runs
                }

                param_logging("after_unsup_finetune", params);
            }
            ////////////////////////////////////////////////////////////
            //                 supervised finetuning
            ////////////////////////////////////////////////////////////
            if(m_finetune){
                std::cout <<".supervised finetuning"<<std::endl;
                std::vector<Op*> params;
                for(unsigned int l=0; l<m_aes->size(); l++) // derive w.r.t. /all/ parameters except output bias of AEs
                {
                    std::vector<Op*> tmp = m_aes->get(l).supervised_params();
                    std::copy(tmp.begin(), tmp.end(), std::back_inserter(params));
                }
                //params += m_aes->get(l).weights().get(), m_aes->get(l).bias_h().get();
                params += m_mlp->weights().get(), m_mlp->bias().get();

                gradient_descent gd(m_mlp->loss(),0,params,m_mlp_lr,0.00000f);
                gd.before_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
                gd.after_epoch.connect(boost::bind(&pretrained_mlp::log_loss,m_mlp.get(), _1));
                gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_supervised,this,_2));
                gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_loss,m_mlp.get()));
                gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
                gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));

                if(m_sdl.can_earlystop()) {
                    //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
                    gd.setup_early_stopping(boost::bind(&pretrained_mlp::perf,m_mlp.get()), 5, 0.00001f, 2);
                    gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
                    gd.before_validation_epoch.connect(boost::bind(&sdl_t::before_validation_epoch,&m_sdl));
                    gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp_trainer::validation_epoch,this,true));
                    gd.after_validation_epoch.connect(1, boost::bind(&sdl_t::after_validation_epoch,&m_sdl));
                    gd.after_validation_epoch.connect(0,boost::bind(&pretrained_mlp::log_loss, m_mlp.get(), _1));
                    gd.after_validation_epoch.connect(1, boost::bind(&pretrained_mlp_trainer::validation_epoch,this,false));
                    gd.minibatch_learning(10000);
                    m_mlp->s_epochs(gd.rounds()); // remember number of iterations until optimum
                } else {
                    std::cout << "TRAINALL phase: mlp avg_epochs="<<m_mlp->avg_epochs()<<std::endl;
                    gd.minibatch_learning(m_mlp->avg_epochs()); // TRAINALL phase. use as many iterations as in previous runs
                }
                param_logging("after_sup_finetune", params);
            }
        }
        void validation_epoch(bool b){
            g_worker->log(BSON("who"<<"trainer"<<"topic"<<"validation"<<"validation_mode"<<b));
            g_worker->checkpoint();
        }
        void switch_dataset(unsigned int split, cv_mode mode){
            g_worker->log(BSON("who"<<"trainer"<<"topic"<<"switch_dataset"<<"split"<<split<<"mode"<<mode));
            g_worker->checkpoint();
            m_sdl.switch_dataset(split,mode);
        }
        void reset_params(){
            m_mlp->reset_weights();
            m_aes->reset_weights();
        }
        unsigned int n_splits(){
            return m_sdl.n_splits();
        }
    private:
        void load_batch_supervised(unsigned int batch){
            m_aes->input()  = m_sdl.get_data_batch(batch).copy();
            m_mlp->target() = m_sdl.get_label_batch(batch).copy();
        }
        void load_batch_unsupervised(unsigned int batch){
            m_aes->input()  = m_sdl.get_data_batch(batch).copy();
        }
};

BOOST_CLASS_EXPORT(crossvalidatable);
BOOST_CLASS_EXPORT(pretrained_mlp_trainer);

double log_uniform(double vmin, double vmax){
    double r = drand48();
    r *= log(vmax)-log(vmin);
    r += log(vmin);
    return exp(r);
}
double uniform(double vmin, double vmax){
    double r = drand48();
    r *= vmax-vmin;
    r += vmin;
    return r;
}

void generate_and_test_models_random(boost::asio::deadline_timer* dt, boost::asio::io_service* io, cv::crossvalidation_queue* q) {
    size_t n_open     = q->m_hub.get_n_open();
    size_t n_finished = q->m_hub.get_n_ok();
    size_t n_assigned = q->m_hub.get_n_assigned();
    std::cout << "o:"<<n_open<<" f:"<<n_finished<<" a:"<<n_assigned<<" x:"<<q->m_hub.get_n_failed()<<std::endl;
    if(n_open<3){
        boost::shared_ptr<crossvalidatable> p(new pretrained_mlp_trainer());
        std::cout <<"generating new sample"<<std::endl;

        //unsigned int n_layers = 1+3*drand48();
        unsigned int n_layers = 2;

        float mlp_lr  = log_uniform(0.01, 0.2);
        float aes_lr0  = log_uniform(0.01, 0.2);
        //float mlp_lr  = 0.1;
        //float aes_lr0  = 0.1;
        std::vector<float> lambda(n_layers);
        std::vector<float> aes_lr(n_layers);
        std::vector<float> noise(n_layers);
        std::vector<int  > size(n_layers);
        std::vector<bool > twolayer(n_layers);

        float lambda0 = log_uniform(0.0001, 1.0);
        if(drand48()<0.1)
            lambda0 = 0.f;

        for (unsigned int i = 0; i < n_layers; ++i)
        {
            lambda[i] = lambda0;
            aes_lr[i] = aes_lr0;
            noise[i]  = 0.0;
            size[i]   = 
                (i==0 ? 512 : 256 );
                //512;
                //int(pow(28+drand48()*8,2));
                //((i==0) ? 5*30 : 15);// hidden0: 4*message plus message, hidden1: only message
            twolayer[i] = (i<n_layers-1);
        }

        std::string uuid = boost::lexical_cast<std::string>(boost::uuids::uuid(boost::uuids::random_generator()()));
        for (int idx0 = 0; idx0 < 3; ++idx0)
        {
            mongo::BSONObjBuilder bob;
            bob << "uuid" << uuid;
            bob << "dataset" << "msrc_descr";
            bob << "bs"      << 128;
            bob << "nsplits" << 1;
            bob << "mlp_lr"  << mlp_lr;

            bob << "pretrain" << (drand48()>0.1f);
            bob << "ufinetune" << true;
            bob << "sfinetune" << false;

            if(idx0 == 2){
                n_layers = 1;
                size[0] = size[1]; // skip 1st layer
            }

            mongo::BSONArrayBuilder stack;
            for (unsigned int i = 0; i < n_layers; ++i)
            {
                stack << BSON(
                        "lambda"   << lambda[i]   <<
                        "lr"       << aes_lr[i]   <<
                        "noise"    << noise[i]    <<
                        "size"     << size[i]     <<
                        // exactly same settings, but w/ and w/o twolayer
                        "twolayer" << ((idx0==0) ? true : false)
                        );
            }
            bob << "stack"<<stack.arr();
            q->dispatch(p, bob.obj());
        }
    }

    dt->expires_at(dt->expires_at() + boost::posix_time::seconds(1));
    dt->async_wait(boost::bind(generate_and_test_models_random, dt, io, q));
}

void generate_and_test_models_test(boost::asio::deadline_timer* dt, boost::asio::io_service* io, cv::crossvalidation_queue* q) {
    size_t n_open     = q->m_hub.get_n_open();
    size_t n_finished = q->m_hub.get_n_ok();
    size_t n_assigned = q->m_hub.get_n_assigned();
    std::cout << "o:"<<n_open<<" f:"<<n_finished<<" a:"<<n_assigned<<" x:"<<q->m_hub.get_n_failed()<<std::endl;
    boost::shared_ptr<crossvalidatable> p(new pretrained_mlp_trainer());
    q->dispatch( p,
            BSON(
                "dataset"     << "natural"   <<
                "bs"          << 16          <<
                "nsplits"     << 1           <<
                "mlp_lr"      << 0.1         <<
                "pretrain"    << true        <<
                "ufinetune"   << false       <<
                "sfinetune"   << false       <<
                "stack"       << BSON_ARRAY(
                    BSON(
                        "lambda"   << 1.0        <<
                        "lr"       << 0.1        <<
                        "noise"    << 0.0        <<
                        "size"     << 81         <<
                        "twolayer" << false))));
    q->dispatch( p,
            BSON(
                "dataset"   << "natural"   <<
                "bs"        << 16          <<
                "nsplits"   << 1           <<
                "mlp_lr"    << 0.1         <<
                "pretrain"  << true        <<
                "ufinetune" << false       <<
                "sfinetune" << false       <<
                "stack"     << BSON_ARRAY(
                    BSON(
                        "lambda"   << 1.0        <<
                        "lr"       << 0.1        <<
                        "noise"    << 0.0        <<
                        "size"     << 81         <<
                        "twolayer" << false)
                    <<
                    BSON(
                        "lambda"   << 1.0        <<
                        "lr"       << 0.1        <<
                        "noise"    << 0.0        <<
                        "size"     << 81         <<
                        "twolayer" << false))));

    dt->expires_at(dt->expires_at() + boost::posix_time::seconds(1));
    dt->async_wait(boost::bind(generate_and_test_models_test, dt, io, q));
}


int main(int argc, char **argv)
{

    srand48(time(NULL));
    if(argc<=1){
        std::cout <<"Usage: "<<argv[0] << " {hub|client|test} "<<std::endl;
        std::cout <<"   - hub    args:  "<<std::endl;
        std::cout <<"   - client args: {device} "<<std::endl;
        std::cout <<"   - test: {device} "<<std::endl;
        return 1;
    }

    const std::string hc_db = "test.msrc";
    boost::asio::io_service io;
    if(std::string("hub") == argv[1]){
        cv::crossvalidation_queue q("131.220.7.92",hc_db);
        std::cout << "Clear database `"<<hc_db<<"'? --> type `yes'" << std::endl;
        std::string s;
        std::cin >> s;
        if(s=="yes"){
            std::cout << "clearing....." << std::endl;
            q.m_hub.clear_all();
        }
        boost::asio::deadline_timer dt(io, boost::posix_time::seconds(1));
        dt.async_wait(boost::bind(generate_and_test_models_random, &dt, &io, &q));

        q.m_hub.reg(io,1); // checks for timeouts
        io.run();
    }
    if(std::string("client") == argv[1]){
        cuvAssert(argc==3);
        cuv::initCUDA(boost::lexical_cast<int>(argv[2]));
        cuv::initialize_mersenne_twister_seeds(time(NULL));
        cv::crossvalidation_worker w("131.220.7.92",hc_db);
        w.reg(io,1);
        io.run();
    }
    if(std::string("test") == argv[1]){
        cuvAssert(argc==3);
        cuv::initCUDA(boost::lexical_cast<int>(argv[2]));
        cuv::initialize_mersenne_twister_seeds(time(NULL));

        cv::crossvalidation_queue q("131.220.7.92","test.dev");
        cv::crossvalidation_worker w("131.220.7.92","test.dev");

        boost::asio::deadline_timer dt(io, boost::posix_time::seconds(1));
        dt.async_wait(boost::bind(generate_and_test_models_random, &dt, &io, &q));
        q.m_hub.clear_all();

        q.m_hub.reg(io,1); // checks for timeouts
        w.reg(io,1);

        io.run();
    }

    return 0;
}

