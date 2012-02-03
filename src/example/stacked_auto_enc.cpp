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

#include <cuvnet/op_io.hpp>
#include <tools/crossvalid.hpp>

using namespace cuvnet;
using namespace boost::assign;
using boost::make_shared;

typedef boost::shared_ptr<Input>  input_ptr;
typedef boost::shared_ptr<Output> output_ptr;
typedef boost::shared_ptr<Op>     op_ptr;

class learn_base {
};

/**
 * one layer of a stacked auto encoder
 */
struct auto_encoder {
    private:
        op_ptr       m_input;
        input_ptr    m_weights,m_bias_h,m_bias_y;
        output_ptr   m_loss_sink;
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
            std::cout << "AE"<<cuv::getCurrentDevice()<<" "<< epoch<<" " << m_loss_sum/m_loss_sum_cnt<<std::endl;
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
                ar &  m_aes & m_params;
            }
        std::string m_params;
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
        auto_enc_stack(unsigned int bs  , unsigned int inp1, int n_layers, const int* layer_sizes, bool binary, float noise=0.0f, float lambda=0.0f) {
            m_aes.push_back(auto_encoder(bs  ,inp1,layer_sizes[0], binary,noise,lambda));
            // TODO: do not use noise in 1st layer when training 2nd layer
            std::ostringstream ss;
            ss<<"{\"AES\": { \"bs\":"<<bs<<", \"inp1\":"<<inp1<<", \"n_layers\":"<<n_layers<<", \"layers\": [" <<layer_sizes[0];
            for(int i=1; i<n_layers; i++) {
                op_ptr out = m_aes.back().output();
                m_aes.push_back(auto_encoder(i,out,layer_sizes[i],true,noise,lambda));
                ss<<", "<<layer_sizes[i];
            }
            ss<<"], \"binary\":"<<binary<<", \"noise\":"<<noise<<", \"lambda\":"<<lambda<<" }}";
            m_params = ss.str();
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
        output_ptr m_out_sink;
        output_ptr m_loss_sink;
        unsigned int m_epochs;///< number of epochs this was trained for
        float m_loss_sum, m_class_err;
        unsigned int m_loss_sum_cnt, m_class_err_cnt;

        friend class boost::serialization::access;
        friend std::ostream& operator<<(std::ostream&, const pretrained_mlp&);
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) {
                ar &  m_params & m_input & m_loss & m_targets & m_weights & m_bias & m_output & m_out_sink & m_loss_sink & m_loss_sum & m_class_err & m_loss_sum_cnt & m_class_err_cnt & m_epochs;
            }
        std::string m_params;
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
            std::cout <<"MLP"<<cuv::getCurrentDevice()<<": "<< epoch;
            if(m_loss_sum_cnt && m_class_err_cnt)
                std::cout << " loss: "<<m_loss_sum/m_loss_sum_cnt << ", ";
            if(m_class_err_cnt)
                std::cout << " cerr: "<<m_class_err/m_class_err_cnt;
            std::cout<<std::endl;
        }
        output_ptr output() {
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

                std::ostringstream ss;
                ss << "{ \"MLP\" : { \"bs\" : "<<bs<<",  \"inp1\":"<<inp1<<", \"softmax\":"<<softmax <<" }}";

                // create a sink for outputs so we can determine classification error
                m_out_sink = sink(m_output);
                if(softmax) // multinomial logistic regression
                    m_loss = mean(multinomial_logistic_loss(m_output, m_targets,1));
                else        // mean squared error
                    m_loss = mean(pow(axpby(-1.f,m_targets,m_output),2.f));
                m_loss_sink = sink(m_loss);
                m_params = ss.str();
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
std::ostream& operator<<(std::ostream& o, const auto_enc_stack& aes)
{
    o << aes.m_params;
    return o;
}
std::ostream& operator<<(std::ostream& o, const pretrained_mlp& mlp)
{
    o << mlp.m_params;
    return o;
}

/**
 * contains fit and predict functions for a pretrained MLP
 */
struct pretrained_mlp_trainer {
    boost::shared_ptr<pretrained_mlp> m_mlp; ///< the mlp to be trained
    boost::shared_ptr<auto_enc_stack> m_aes; ///< the stacked ae to be pre-trained

    Op::value_type m_current_data;  ///< should contain the dataset we're working on
    Op::value_type m_current_labels; ///< should contain the labels of the dataset we're working on
    Op::value_type m_current_vdata;  ///< should contain the validation dataset we're working on
    Op::value_type m_current_vlabels; ///< should contain the validation labels of the dataset we're working on
    unsigned int m_bs;
    bool m_in_validation_mode;
    bool m_pretraining; ///< whether we want pretraining
    float m_lr_ae; ///< learning rate of AE
    float m_lr_mlp; ///< learning rate of MLP

    /**
     * constructor
     * @param mlp the mlp where params should be learned
     */
    pretrained_mlp_trainer(auto_enc_stack* aes, pretrained_mlp* mlp, bool pretrain, float lr_ae, float lr_mlp)
        :m_mlp(mlp), m_aes(aes), m_bs(32), m_in_validation_mode(false), m_pretraining(pretrain), m_lr_ae(lr_ae), m_lr_mlp(lr_mlp) {}

    void before_validation_epoch() {
        m_in_validation_mode = true;
    }
    void after_validation_epoch() {
        m_in_validation_mode = false;
    }

    /**
     * default constructor for convenience
     */
    pretrained_mlp_trainer() {}

    /**
     * load a batch in an autoencoder
     * @param ae    the autoencoder
     * @param data  the source dataset
     * @param bs    the size of one batch
     * @param batch the number of the requested batch
     */
    void load_batch_ae(
        auto_encoder* ae, unsigned int batch) {
        Op::value_type& data = m_in_validation_mode ? m_current_vdata : m_current_data;
        ae->input() = data[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
    }

    /**
     * load a batch in an MLP
     * @param ae    the autoencoder (where we need to put the inputs)
     * @param mlp   the mlp (where we need to put the targets)
     * @param data    the source dataset inputs
     * @param labels  the source dataset labels
     * @param bs    the size of one batch
     * @param batch the number of the requested batch
     */
    void load_batch_mlp(
        auto_encoder* ae, pretrained_mlp* mlp, unsigned int batch) {
        Op::value_type& data = m_in_validation_mode ? m_current_vdata : m_current_data;
        Op::value_type& labl = m_in_validation_mode ? m_current_vlabels : m_current_labels;
        ae->input()   = data[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
        mlp->target() = labl[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
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
        gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_mlp,this,&m_aes->get(0),m_mlp.get(),_2));
        gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
        gd.current_batch_num.connect(boost::bind(&pretrained_mlp_trainer::current_batch_num,this));
        gd.minibatch_learning(1,0,0); // 1 epoch
        return m_mlp->perf();
    }

    /**
     * train the given auto_encoder stack and the mlp
     */
    void fit() {
        ////////////////////////////////////////////////////////////
        //             un-supervised pre-training
        ////////////////////////////////////////////////////////////
        if(m_pretraining)
            for(unsigned int l=0; l<m_aes->size(); l++) {
                std::vector<Op*> params;
                params += m_aes->get(l).weights().get(), m_aes->get(l).bias_y().get(), m_aes->get(l).bias_h().get();

                gradient_descent gd(m_aes->get(l).loss(),0,params,m_lr_ae,0.00000f);
                gd.before_epoch.connect(boost::bind(&auto_encoder::reset_loss, &m_aes->get(l)));
                gd.after_epoch.connect(boost::bind(&auto_encoder::print_loss, &m_aes->get(l), _1));
                gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_ae,this,&m_aes->get(0),_2));
                gd.after_batch.connect(boost::bind(&auto_encoder::acc_loss,&m_aes->get(l)));
                gd.current_batch_num.connect(boost::bind(&pretrained_mlp_trainer::current_batch_num,this));

                if(m_current_vdata.ptr()) {
                    // we can only use early stopping when validation data is given
                    //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
                    gd.setup_early_stopping(boost::bind(&auto_encoder::perf,&m_aes->get(l)), 5, 0.1f, 2);
                    gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp_trainer::before_validation_epoch,this));
                    gd.after_validation_epoch.connect( boost::bind(&pretrained_mlp_trainer::after_validation_epoch,this));
                    gd.minibatch_learning(10000);
                } else {
                    std::cout << "TRAINALL phase: aes"<<l<<" epochs="<<m_aes->get(l).epochs()<<std::endl;
                    gd.minibatch_learning(m_aes->get(l).epochs()); // TRAINALL phase. Use as many as in previous runs
                }
            }
        ////////////////////////////////////////////////////////////
        //                 supervised training
        ////////////////////////////////////////////////////////////
        std::vector<Op*> params;
        for(unsigned int l=0; l<m_aes->size(); l++) // derive w.r.t. /all/ parameters except output bias of AEs
            params += m_aes->get(l).weights().get(), m_aes->get(l).bias_h().get();
        params += m_mlp->weights().get(), m_mlp->bias().get();

        gradient_descent gd(m_mlp->loss(),0,params,m_lr_mlp,0.00000f);
        gd.before_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
        gd.after_epoch.connect(boost::bind(&pretrained_mlp::print_loss,m_mlp.get(), _1));
        gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_mlp,this,&m_aes->get(0),m_mlp.get(),_2));
        gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_loss,m_mlp.get()));
        gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
        gd.current_batch_num.connect(boost::bind(&pretrained_mlp_trainer::current_batch_num,this));

        if(m_current_vdata.ptr()) {
            //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
            gd.setup_early_stopping(boost::bind(&pretrained_mlp::perf,m_mlp.get()), 5, 0.001f, 2);
            gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp_trainer::before_validation_epoch,this));
            gd.after_validation_epoch.connect( boost::bind(&pretrained_mlp_trainer::after_validation_epoch,this));
            gd.minibatch_learning(10000);
        } else {
            std::cout << "TRAINALL phase: mlp epochs="<<m_mlp->epochs()<<std::endl;
            gd.minibatch_learning(m_mlp->epochs()); // TRAINALL phase. use as many iterations as in previous runs
        }
    }
    unsigned int current_batch_num() {
        return  m_in_validation_mode ?
                m_current_vdata.shape(0)/m_bs :
                m_current_data.shape(0)/m_bs;
    }
    std::string desc() {
        std::ostringstream ss;
        ss << *m_aes <<", "<< *m_mlp <<", {\"trainer\":{\"pretrain\":"<<m_pretraining << ",\"lr_ae\":"<<m_lr_ae<<",\"lr_mlp\":"<<m_lr_mlp<<"}}";
        return ss.str();
    }
    void reset_params() {
        std::cout << "Parameter reset!"<<std::endl;
        m_aes->reset_weights();
        m_mlp->reset_weights();
    }
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar &  m_mlp & m_aes & m_bs & m_in_validation_mode & m_pretraining & m_lr_ae & m_lr_mlp;
    }

};


void prepare_ds(pretrained_mlp_trainer* pmt, splitter* splits, unsigned int split, crossvalidation<pretrained_mlp_trainer>::cv_mode cm)
{
    std::cout << "----------------> switching to split "<<split<<", cm:"<<cm<<std::endl;
    cuv::tensor<float,cuv::host_memory_space> data, vdata;
    cuv::tensor<int,  cuv::host_memory_space> labels, vlabels;
    dataset ds = (*splits)[split];
    switch(cm) {
    case crossvalidation<pretrained_mlp_trainer>::CM_TRAINALL:
        data   = splits->get_ds().train_data;
        labels = splits->get_ds().train_labels;
        break;
    case crossvalidation<pretrained_mlp_trainer>::CM_TRAIN:
        data   = ds.train_data;
        labels = ds.train_labels;
        vdata   = ds.val_data;      // for early stopping!
        vlabels = ds.val_labels;    // for early stopping!
        break;
    case crossvalidation<pretrained_mlp_trainer>::CM_VALID:
        data   = ds.val_data;
        labels = ds.val_labels;
        break;
    case crossvalidation<pretrained_mlp_trainer>::CM_TEST:
        data   = ds.test_data;
        labels = ds.test_labels;
        break;
    };

    // convert labels to float
    cuv::tensor<float,cuv::host_memory_space> flabels(labels.shape());
    cuv::convert(flabels,  labels);
    pmt->m_current_data    = data;
    pmt->m_current_labels  = flabels;
    if(vlabels.ndim()) {
        cuv::tensor<float,cuv::host_memory_space> fvlabels(vlabels.shape());
        cuv::convert(fvlabels, vlabels);
        pmt->m_current_vdata   = vdata;
        pmt->m_current_vlabels = fvlabels;
    } else {
        pmt->m_current_vdata.dealloc();
        pmt->m_current_vlabels.dealloc();
    }
}

class pmlp_cv
        : public crossvalidation<pretrained_mlp_trainer> {
private:
    dataset& m_ds;
    splitter m_splits;
public:
    pmlp_cv(dataset& ds, unsigned int splits, unsigned int workers, unsigned int startdev)
        : crossvalidation<pretrained_mlp_trainer>(splits,workers,startdev)
        , m_ds(ds)
        , m_splits(ds, splits) {
        this->switch_dataset.connect(boost::bind(prepare_ds,_1,&m_splits,_2,_3));
    }
    void generate_and_test_models() {
        unsigned int bs=32;
        for(unsigned int i=0; i<1000; i++) {
            float lambda = drand48();
            unsigned int n_layers = 1+3*drand48();
            bool pretrain = drand48()>0.2f;
            int fn = 28 + drand48()*8;
            int layer_size[]  = {fn*fn, fn*fn, fn*fn, fn*fn};
            float lr_mlp = 0.01f + drand48()*0.2f;
            float lr_ae  = 0.01f + drand48()*0.2f;
            auto_enc_stack* aes = new auto_enc_stack( bs,
                    m_ds.train_data.shape(1),
                    n_layers, layer_size,
                    m_ds.channels==1, 0.00f, lambda);
            pretrained_mlp* mlp = new pretrained_mlp(
                aes->get(n_layers-1).output(), 10, true);
            pretrained_mlp_trainer pmlpt(aes, mlp,pretrain, lr_ae, lr_mlp);  // takes ownership of aes and mlp
            dispatch(pmlpt);
        }
    }
};

int main(int argc, char **argv)
{
    srand48(time(NULL));
    cuvAssert(argc==4);
    cudaSetDevice(boost::lexical_cast<int>(argv[1]));
    unsigned int ndev = boost::lexical_cast<int>(argv[2]);
    unsigned int startdev = boost::lexical_cast<int>(argv[3]);
    cuv::initialize_mersenne_twister_seeds(time(NULL));
    std::cout << "main: on device "<<cuv::getCurrentDevice()<<std::endl;

    mnist_dataset ds_all("/home/local/datasets/MNIST");
    global_min_max_normalize<> normalizer(0,1); // 0,1
    //cifar_dataset ds;
    //zero_mean_unit_variance<> normalizer;
    //amat_dataset ds_all("/home/local/datasets/bengio/mnist.zip","mnist_train.amat", "mnist_test.amat");
    //global_min_max_normalize<> normalizer(0,1); // 0,1
    dataset ds = randomizer().transform(ds_all);

    normalizer.fit_transform(ds.train_data);
    normalizer.transform(ds.test_data);

    pmlp_cv trainer(ds, 3,ndev, startdev);
    float best_result = trainer.run();
    std::cout << "best result: "<<best_result<<std::endl;

    return 0;
}

