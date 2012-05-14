// vim:ts=4:sw=4:et
#include <signal.h>
#include <fstream>
#include <cmath>
#include <boost/assign.hpp>
#include <boost/bind.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <cuvnet/ops.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <tools/visualization.hpp>
#include <tools/preprocess.hpp>
#include <tools/gradient_descent.hpp>
#include <datasets/cifar.hpp>
#include <datasets/mnist.hpp>
#include <datasets/amat_datasets.hpp>
#include <datasets/splitter.hpp>
#include <datasets/randomizer.hpp>
#include <mongo/bson/bson.h>

#include <boost/serialization/export.hpp>
#include <boost/serialization/export.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <cuvnet/op_io.hpp>
#include <tools/crossvalid.hpp>
#include <tools/learner.hpp>

namespace cuvnet{
    extern cv::crossvalidation_worker* g_worker;
}

using namespace cuvnet;
using namespace boost::assign;

typedef boost::shared_ptr<Input>  input_ptr;
typedef boost::shared_ptr<Sink> sink_ptr;
typedef boost::shared_ptr<Op>     op_ptr;

/**
 * one layer of a stacked auto encoder
 */
struct auto_encoder {
    private:
        op_ptr       m_input;
        input_ptr    m_weights,m_bias_h,m_bias_y;
        sink_ptr   m_loss_sink;
        op_ptr       m_decode, m_enc;
        op_ptr       m_loss, m_rec_loss, m_contractive_loss;
        float        m_loss_sum;
        unsigned int m_epochs; ///< number of epochs this was trained for TODO: reset this together with reset_params and/or count how many times it was reset to get the average!
        unsigned int m_loss_sum_cnt;

        friend class boost::serialization::access;
        friend std::ostream& operator<<(std::ostream&, const auto_encoder&);
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ar &  m_input & m_weights & m_bias_h & m_bias_y 
                    & m_decode & m_enc & m_loss & m_rec_loss & m_loss_sink
                    & m_contractive_loss & m_loss_sum & m_loss_sum_cnt &  m_epochs;
            }
    public:
        input_ptr       weights() { return m_weights; }
        input_ptr       bias_h()  { return m_bias_h; }
        input_ptr       bias_y()  { return m_bias_y; }
        op_ptr          loss()    { return m_loss; }
        unsigned int    epochs()  { return m_epochs; }

        matrix&       input() {
            return boost::dynamic_pointer_cast<Input>(m_input)->data();
        }
        op_ptr output() {
            return m_enc;
        }
        void acc_loss() {
            m_loss_sum += m_loss_sink->cdata()[0];
            m_loss_sum_cnt ++;
        }
        void reset_loss() {
            m_loss_sum = m_loss_sum_cnt = 0;
        }
        void print_loss(unsigned int epoch) {
            m_epochs = std::max(epoch, m_epochs);
            g_worker->log(BSON("who"<<"AE"<<"epoch"<<epoch<<"perf"<<m_loss_sum/m_loss_sum_cnt));
            g_worker->checkpoint();
        }
        float perf() {
            return m_loss_sum/m_loss_sum_cnt;
        }

        auto_encoder() {} ///< default ctor for serialization
        /**
         * this constructor gets the \e encoded output of another autoencoder as
         * its input and infers shapes from there.
         *
         * @param layer the number of the layer (used for naming the parameters)
         * @param inputs the "incoming" (=lower level) encoded representation
         * @param hl   size of encoded representation
         * @param binary if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_encoder(unsigned int layer, op_ptr& inputs, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
            :m_input(inputs)
             ,m_bias_h(new Input(cuv::extents[hl],       "ae_bias_h"+ boost::lexical_cast<std::string>(layer))) {
                 m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
                 unsigned int bs   = inputs->result()->shape[0];
                 unsigned int inp1 = inputs->result()->shape[1];
                 m_weights.reset(new Input(cuv::extents[inp1][hl],"ae_weights" + boost::lexical_cast<std::string>(layer)));
                 m_bias_y.reset(new Input(cuv::extents[inp1],     "ae_bias_y"  + boost::lexical_cast<std::string>(layer)));
                 m_epochs=0;
                 init(bs  ,inp1,hl,binary,noise,lambda);
             }

        /** this constructor is used for the outermost autoencoder in a stack
         * @param bs   batch size
         * @param inp1 number of variables in one pattern
         * @param hl   size of encoded representation
         * @param binary if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_encoder(unsigned int bs  , unsigned int inp1, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
            :m_input(new Input(cuv::extents[bs  ][inp1],"ae_input"))
             ,m_weights(new Input(cuv::extents[inp1][hl],"ae_weights"))
             ,m_bias_h(new Input(cuv::extents[hl],       "ae_bias_h"))
             ,m_bias_y(new Input(cuv::extents[inp1],     "ae_bias_y")) {
                 init(bs  ,inp1,hl,binary,noise,lambda);
             }

        /** 
         * Initializes weights and biases
         * 
         */    
        void reset_weights() {
            // initialize weights and biases
            float wnorm = m_weights->data().shape(0)
                +         m_weights->data().shape(1);
            float diff = 4.f*std::sqrt(6.f/wnorm);
            cuv::fill_rnd_uniform(m_weights->data());
            m_weights->data() *= 2*diff;
            m_weights->data() -=   diff;
            m_bias_h->data()   = 0.f;
            m_bias_y->data()   = 0.f;
            m_epochs           = 0;
        }

    private:
        /**
         * initializes the functions in the AE  according to params given in the
         * constructor
         */
        void init(unsigned int bs  , unsigned int inp1, unsigned int hl, bool binary, float noise, float lambda) {
            Op::op_ptr corrupt               = m_input;
            if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
            if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);
            m_enc    = logistic(mat_plus_vec(
                        prod( corrupt, m_weights)
                        ,m_bias_h,1));
            m_decode = mat_plus_vec(
                    prod( m_enc, m_weights, 'n','t')
                    ,m_bias_y,1);

            if(!binary)  // squared loss
                m_rec_loss = mean( pow( axpby(m_input, -1.f, m_decode), 2.f));
            else         // cross-entropy
                m_rec_loss = mean( sum(neg_log_cross_entropy_of_logistic(m_input,m_decode),1));

            if(lambda>0.f) { // contractive AE
                m_contractive_loss =
                    sum(sum(pow(m_enc*(1.f-m_enc),2.f),0)
                            * sum(pow(m_weights,2.f),0));
                m_loss        = axpby(m_rec_loss, lambda/(float)bs  , m_contractive_loss);
            } else
                m_loss        = m_rec_loss; // no change
            m_loss_sink       = sink(m_loss);
            reset_weights();
        }
};

/**
 * a stack of multiple `auto_encoder's.
 */
struct auto_enc_stack {
    private:
        std::vector<auto_encoder> m_aes; ///< all auto encoders
        friend class boost::serialization::access;
        friend std::ostream& operator<<(std::ostream&, const auto_enc_stack&);
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ar &  m_aes;
            }
    public:
        auto_enc_stack() {} ///< default ctor for serialization

        /**
         * construct an auto-encoder stack
         *
         * @param bs       batch size
         * @param inp1     size of input layer
         * @param n_layers "height" of the stack
         * @param layer_sizes an array of n_layers integers denoting the "hidden" layer sizes
         * @param binary   if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_enc_stack(unsigned int bs  , unsigned int inp1, int n_layers, const int* layer_sizes, bool binary, float* noise, float* lambdas) {
            m_aes.push_back(auto_encoder(bs  ,inp1,layer_sizes[0], binary,noise[0],lambdas[0]));
            // TODO: do not use noise in 1st layer when training 2nd layer
            for(int i=1; i<n_layers; i++) {
                op_ptr out = m_aes.back().output();
                m_aes.push_back(auto_encoder(i,out,layer_sizes[i],true,noise[i],lambdas[i]));
            }
        }
        /**
         * get a specific auto encoder
         * @param i the layer number of the AE
         * @return the i-th AE
         */
        auto_encoder& get(unsigned int i) {
            return m_aes[i];
        }
        matrix&        input() {
            return m_aes.front().input();
        }
        op_ptr output() {
            return m_aes.back().output();
        }
        unsigned int size()const{
            return m_aes.size();
        }
        void reset_weights() {
            for(unsigned int i=0; i<m_aes.size(); i++) {
                m_aes[i].reset_weights();
            }
        }

};

/**
 * for supervised optimization of an objective
 */
struct pretrained_mlp {
    private:
        op_ptr    m_input, m_loss;
        input_ptr m_targets;
        input_ptr m_weights;
        input_ptr m_bias;
        op_ptr    m_output; ///< classification result
        sink_ptr m_out_sink;
        sink_ptr m_loss_sink;
        unsigned int m_epochs;///< number of epochs this was trained for
        float m_loss_sum, m_class_err;
        unsigned int m_loss_sum_cnt, m_class_err_cnt;

        friend class boost::serialization::access;
        friend std::ostream& operator<<(std::ostream&, const pretrained_mlp&);
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ar &  m_input & m_loss & m_targets & m_weights & m_bias & m_output & m_out_sink & m_loss_sink & m_loss_sum & m_class_err & m_loss_sum_cnt & m_class_err_cnt & m_epochs;
            }
    public:

        unsigned int    epochs()  { return m_epochs; }
        input_ptr      weights()  { return m_weights; }
        input_ptr      bias   ()  { return m_bias; }

        Op::value_type& target() {
            return m_targets->data();
        }
        void acc_loss()     {
            m_loss_sum += m_loss_sink->cdata()[0];
            m_loss_sum_cnt++;
        }
        void acc_class_err() {
            cuv::tensor<int,Op::value_type::memory_space_type> a1 ( m_out_sink->cdata().shape(0) );
            cuv::tensor<int,Op::value_type::memory_space_type> a2 ( m_targets->data().shape(0) );
            cuv::reduce_to_col(a1, m_out_sink->cdata(),cuv::RF_ARGMAX);
            cuv::reduce_to_col(a2, m_targets->data(),cuv::RF_ARGMAX);
            m_class_err     += m_out_sink->cdata().shape(0) - cuv::count(a1-a2,0);
            m_class_err_cnt += m_out_sink->cdata().shape(0);
        }
        float perf() {
            return m_class_err/m_class_err_cnt;
        }
        void reset_loss() {
            m_loss_sum = m_class_err = m_class_err_cnt = m_loss_sum_cnt = 0;
        }
        void print_loss(unsigned int epoch) {
            m_epochs = std::max(epoch, m_epochs);
            mongo::BSONObjBuilder bob;
            bob<<"who"<<"mlp"<<"epoch"<<epoch;
            if(m_loss_sum_cnt && m_class_err_cnt){
                bob<<"loss"<<m_loss_sum/m_loss_sum_cnt;
            }
            if(m_class_err_cnt){
                bob<<"cerr"<<m_class_err/m_class_err_cnt;
            }
            g_worker->log(bob.obj());
            g_worker->checkpoint();
        }
        sink_ptr output() {
            return m_out_sink;
        }
        op_ptr loss(){ return m_loss; }

        pretrained_mlp() {} ///< default ctor for serialization

        /**
         * constructor
         *
         * @param inputs  where data is coming from
         * @param outputs number of targets per input (the matrix created will be of size inputs.shape(0) times outputs)
         * @param softmax if true, use softmax for training
         */
        pretrained_mlp(op_ptr inputs, unsigned int outputs, bool softmax)
            :m_input(inputs) {
                m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
                unsigned int bs   = inputs->result()->shape[0];
                unsigned int inp1 = inputs->result()->shape[1];
                m_weights.reset(new Input(cuv::extents[inp1][outputs],"mlp_weights"));
                m_bias.reset   (new Input(cuv::extents[outputs],      "mlp_bias"));
                m_targets.reset(new Input(cuv::extents[bs][outputs], "mlp_target"));
                m_output = mat_plus_vec( prod( m_input, m_weights) ,m_bias,1);

                // create a sink for outputs so we can determine classification error
                m_out_sink = sink(m_output);
                if(softmax) // multinomial logistic regression
                    m_loss = mean(multinomial_logistic_loss(m_output, m_targets,1));
                else        // mean squared error
                    m_loss = mean(pow(axpby(-1.f,m_targets,m_output),2.f));
                m_loss_sink = sink(m_loss);
                m_epochs=0;

                reset_weights();
            }
        /// initialize weights and biases
        void reset_weights() {
            float wnorm = m_weights->data().shape(0)
                +         m_weights->data().shape(1);
            float diff = 4.f*std::sqrt(6.f/wnorm);
            cuv::fill_rnd_uniform(m_weights->data());
            m_weights->data() *= 2*diff;
            m_weights->data() -=   diff;
            m_bias->data()   = 0.f;
            m_bias->data()   = 0.f;
            m_epochs         = 0;
        }
};

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

            m_sdl.constructFromBSON(o);

            unsigned int bs = m_sdl.batchsize();
            unsigned int dd = m_sdl.datadim();

            std::vector<mongo::BSONElement> ar = o["stack"].Array();
            std::vector<int>   layer_sizes(ar.size());
            std::vector<float> noise(ar.size());
            std::vector<float> lambda(ar.size());
            m_aes_lr.resize(ar.size());

            for(unsigned int i=0;i<ar.size(); i++){
                layer_sizes[i] = ar[i].Obj()["size"].Int();
                noise[i]       = ar[i].Obj()["noise"].Double();
                lambda[i]      = ar[i].Obj()["lambda"].Double();
                m_aes_lr[i]    = ar[i].Obj()["lr"].Double();
            }
            bool binary = m_sdl.get_ds().channels == 1; // MNIST vs. CIFAR... TODO: need "binary" property directly in ds
            m_aes.reset(
                new auto_enc_stack(bs,dd,ar.size(),&layer_sizes[0], binary, &noise[0], &lambda[0]));
            m_mlp.reset(
                new pretrained_mlp(m_aes->output(),10, true)); // TODO: fixed number of 10 classes!!???
            m_mlp_lr = o["mlp_lr"].Double();
            m_pretraining = o["pretrain"].Bool();
        }


        /**
         * returns classification error on current dataset
         */
        float predict() {
            // "learning" with learnrate 0 and no weight updates
            std::vector<Op*> params;
            gradient_descent gd(m_mlp->output(),0,params,0.0f,0.0f);
            gd.before_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
            gd.after_epoch.connect(boost::bind(&pretrained_mlp::print_loss, m_mlp.get(),_1));
            gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_supervised,this,_2));
            gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
            gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));
            gd.minibatch_learning(1,0,0); // 1 epoch
            return m_mlp->perf();
        }
        void log_params(std::string desc, std::vector<Op*> ops){
            for(unsigned int i=0;i<ops.size(); i++)
                log_param(desc, ops[i]);
        }
        void log_param(std::string desc, Op* op){
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
                    g_worker->log(BSON("who"<<"trainer"<<"topic"<<"layer_change"<<"layer"<<l));
                    std::vector<Op*> params;
                    params += m_aes->get(l).weights().get(), m_aes->get(l).bias_y().get(), m_aes->get(l).bias_h().get();

                    gradient_descent gd(m_aes->get(l).loss(),0,params,m_aes_lr[l],0.00000f);
                    gd.before_epoch.connect(boost::bind(&auto_encoder::reset_loss, &m_aes->get(l)));
                    gd.after_epoch.connect(boost::bind(&auto_encoder::print_loss, &m_aes->get(l), _1));
                    gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_unsupervised,this,_2));
                    gd.after_batch.connect(boost::bind(&auto_encoder::acc_loss,&m_aes->get(l)));
                    gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));

                    if(m_sdl.can_earlystop()) {
                        // we can only use early stopping when validation data is given
                        //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
                        gd.setup_early_stopping(boost::bind(&auto_encoder::perf,&m_aes->get(l)), 5, 0.1f, 2);
                        gd.before_early_stopping_epoch.connect(boost::bind(&auto_encoder::reset_loss, &m_aes->get(l)));
                        gd.before_early_stopping_epoch.connect(boost::bind(&sdl_t::before_early_stopping_epoch,&m_sdl));
                        gd.before_early_stopping_epoch.connect(boost::bind(&pretrained_mlp_trainer::validation_epoch,this,true));
                        gd.after_early_stopping_epoch.connect(1, boost::bind(&sdl_t::after_early_stopping_epoch, &m_sdl));
                        gd.after_early_stopping_epoch.connect(1, boost::bind(&pretrained_mlp_trainer::validation_epoch,this,false));
                        gd.after_early_stopping_epoch.connect(0, boost::bind(&auto_encoder::print_loss, &m_aes->get(l), _1));
                        gd.minibatch_learning(10000);
                    } else {
                        std::cout << "TRAINALL phase: aes"<<l<<" epochs="<<m_aes->get(l).epochs()<<std::endl;
                        gd.minibatch_learning(m_aes->get(l).epochs()); // TRAINALL phase. Use as many as in previous runs
                    }
                    log_params("after_pretrain", params);
                }
            }
            ////////////////////////////////////////////////////////////
            //                 supervised training
            ////////////////////////////////////////////////////////////
            std::vector<Op*> params;
            for(unsigned int l=0; l<m_aes->size(); l++) // derive w.r.t. /all/ parameters except output bias of AEs
                params += m_aes->get(l).weights().get(), m_aes->get(l).bias_h().get();
            params += m_mlp->weights().get(), m_mlp->bias().get();

            gradient_descent gd(m_mlp->loss(),0,params,m_mlp_lr,0.00000f);
            gd.before_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
            gd.after_epoch.connect(boost::bind(&pretrained_mlp::print_loss,m_mlp.get(), _1));
            gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_supervised,this,_2));
            gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_loss,m_mlp.get()));
            gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
            gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));

            if(m_sdl.can_earlystop()) {
                //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
                gd.setup_early_stopping(boost::bind(&pretrained_mlp::perf,m_mlp.get()), 5, 0.00001f, 2);
                gd.before_early_stopping_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
                gd.before_early_stopping_epoch.connect(boost::bind(&sdl_t::before_early_stopping_epoch,&m_sdl));
                gd.before_early_stopping_epoch.connect(boost::bind(&pretrained_mlp_trainer::validation_epoch,this,true));
                gd.after_early_stopping_epoch.connect(1, boost::bind(&sdl_t::after_early_stopping_epoch,&m_sdl));
                gd.after_early_stopping_epoch.connect(0,boost::bind(&pretrained_mlp::print_loss, m_mlp.get(), _1));
                gd.after_early_stopping_epoch.connect(1, boost::bind(&pretrained_mlp_trainer::validation_epoch,this,false));
                gd.minibatch_learning(10000);
            } else {
                std::cout << "TRAINALL phase: mlp epochs="<<m_mlp->epochs()<<std::endl;
                gd.minibatch_learning(m_mlp->epochs()); // TRAINALL phase. use as many iterations as in previous runs
            }
            log_params("after_train", params);
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
            m_aes->input()  = m_sdl.get_data_batch(batch);
            m_mlp->target() = m_sdl.get_label_batch(batch);
        }
        void load_batch_unsupervised(unsigned int batch){
            m_aes->input()  = m_sdl.get_data_batch(batch);
        }
};

BOOST_CLASS_EXPORT(crossvalidatable);
BOOST_CLASS_EXPORT(pretrained_mlp_trainer);

void generate_and_test_models(boost::asio::deadline_timer* dt, boost::asio::io_service* io, cv::crossvalidation_queue* q) {
    //unsigned int bs=32;
    //for(unsigned int i=0; i<1000; i++) {
        //float lambda = drand48();
        //unsigned int n_layers = 1+3*drand48();
        //bool pretrain = drand48()>0.2f;
        //int fn = 28 + drand48()*8;
        //int layer_size[]  = {fn*fn, fn*fn, fn*fn, fn*fn};
        //q.dispatch(p, desc);
    //}
    size_t n_open     = q->m_hub.get_n_open();
    size_t n_finished = q->m_hub.get_n_ok();
    size_t n_assigned = q->m_hub.get_n_assigned();
    std::cout << "o:"<<n_open<<" f:"<<n_finished<<" a:"<<n_assigned<<std::endl;
    if(n_open<3){
        boost::shared_ptr<crossvalidatable> p(new pretrained_mlp_trainer());
        std::cout <<"generating new sample"<<std::endl;
        mongo::BSONObjBuilder bob;
        bob << "dataset" << "mnist";
        bob << "bs"      << 128;
        bob << "nsplits" << 5;

        bob << "pretrain" << (drand48()>0.2f);

        float mlp_lr  = 0.01f + drand48()*0.2f;
        float aes_lr  = 0.01f + drand48()*0.2f;

        bob << "mlp_lr"<<mlp_lr;

        unsigned int n_layers = 1+3*drand48();
        mongo::BSONArrayBuilder stack;
        for (unsigned int i = 0; i < n_layers; ++i)
        {
            stack << BSON(
                    "lambda"<<drand48() <<
                    "lr"<<aes_lr<<
                    "noise"<<0.0<<
                    "size"<<int(pow(28+drand48()*8,2)));
        }
        bob << "stack"<<stack.arr();
        q->dispatch(p, bob.obj());
    }

    dt->expires_at(dt->expires_at() + boost::posix_time::seconds(1));
    dt->async_wait(boost::bind(generate_and_test_models, dt, io, q));
}


int main(int argc, char **argv)
{
    srand48(time(NULL));
    if(argc<=1){
        std::cout <<"Usage: "<<argv[0] << " {hub|client} "<<std::endl;
        std::cout <<"   - hub    args:  "<<std::endl;
        std::cout <<"   - client args: {device} "<<std::endl;
        return 1;
    }

    boost::asio::io_service io;
    if(std::string("hub") == argv[1]){
        cv::crossvalidation_queue q("localhost","test.mnist");
        //q.m_hub.clear_all();
        boost::asio::deadline_timer dt(io, boost::posix_time::seconds(1));
        dt.async_wait(boost::bind(generate_and_test_models, &dt, &io, &q));

        q.m_hub.reg(io,1); // checks for timeouts
        io.run();
    }
    if(std::string("client") == argv[1]){
        cuvAssert(argc==3);
        cuv::initCUDA(boost::lexical_cast<int>(argv[2]));
        cuv::initialize_mersenne_twister_seeds(time(NULL));
        cv::crossvalidation_worker w("localhost","test.mnist");
        w.reg(io,1);
        io.run();
    }

    return 0;
}

