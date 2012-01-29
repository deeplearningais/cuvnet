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
#include <tools/crossvalid.hpp>
#include <datasets/cifar.hpp>
#include <datasets/mnist.hpp>
#include <datasets/amat_datasets.hpp>
#include <datasets/splitter.hpp>

using namespace cuvnet;
using namespace boost::assign;
using boost::make_shared;

typedef boost::shared_ptr<Input>  input_ptr;
typedef boost::shared_ptr<Output> output_ptr;
typedef boost::shared_ptr<Op>     op_ptr;

/**
 * one layer of a stacked auto encoder
 */
struct auto_encoder{
    op_ptr     m_input;
    input_ptr  m_weights,m_bias_h,m_bias_y;
    output_ptr m_out, m_reconstruct, m_corrupt;
    op_ptr     m_decode, m_enc;
    op_ptr     m_loss, m_rec_loss, m_contractive_loss;
    matrix&       input() {return boost::dynamic_pointer_cast<Input>(m_input)->data();}
    const matrix& output(){return m_out->cdata();}

    float m_loss_sum;
    unsigned int m_loss_sum_cnt;
    void acc_loss(){ m_loss_sum += output()[0]; m_loss_sum_cnt ++; }
    void reset_loss(){m_loss_sum = m_loss_sum_cnt = 0;}
    void print_loss(unsigned int epoch){ std::cout << "AE: "<< epoch<<" " << m_loss_sum/m_loss_sum_cnt<<std::endl;}
    float perf(){ return m_loss_sum/m_loss_sum_cnt; }

    /**
     * this constructor gets the \e encoded output of another autoencoder as
     * its input and infers shapes from there.
     *
     * @param inputs the "incoming" (=lower level) encoded representation
     * @param hl   size of encoded representation
     * @param binary if true, logistic function is applied to outputs
     * @param noise  if noise>0, add noise to the input
     * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
     */
    auto_encoder(op_ptr& inputs, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
        :m_input(inputs)
        ,m_bias_h(new Input(cuv::extents[hl],       "bias_h"))
    {
        m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
        unsigned int bs   = inputs->result()->shape[0];
        unsigned int inp1 = inputs->result()->shape[1];
        m_weights.reset(new Input(cuv::extents[inp1][hl],"weights"));
        m_bias_y.reset(new Input(cuv::extents[inp1],     "bias_y"));
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
    :m_input(new Input(cuv::extents[bs  ][inp1],"input"))
    ,m_weights(new Input(cuv::extents[inp1][hl],"weights"))
    ,m_bias_h(new Input(cuv::extents[hl],       "bias_h"))
    ,m_bias_y(new Input(cuv::extents[inp1],     "bias_y"))
    {init(bs  ,inp1,hl,binary,noise,lambda);}

    private:
    /** 
     * initializes the functions in the AE  according to params given in the
     * constructor
     */
    void init(unsigned int bs  , unsigned int inp1, unsigned int hl, bool binary, float noise, float lambda)
    {
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
        m_out         = make_shared<Output>(m_rec_loss->result()); // reconstruction error
        m_reconstruct = make_shared<Output>(m_decode->result());   // for visualization of reconstructed images
        //m_corrupt = make_shared<Output>(corrupt->result());      // for visualization of corrupted     images

        if(lambda>0.f){ // contractive AE
            m_contractive_loss = 
                sum(sum(pow(m_enc*(1.f-m_enc),2.f),0) 
                        * sum(pow(m_weights,2.f),0));
            m_loss        = axpby(m_rec_loss, lambda/(float)bs  , m_contractive_loss);
        }else
            m_loss        = m_rec_loss; // no change

        // initialize weights and biases
        float diff = 4.f*std::sqrt(6.f/(inp1+hl));
        cuv::fill_rnd_uniform(m_weights->data());
        m_weights->data() *= 2*diff;
        m_weights->data() -=   diff;
        m_bias_h->data()   = 0.f;
        m_bias_y->data()   = 0.f;
    }
};

/**
 * a stack of multiple `auto_encoder's.
 */
struct auto_enc_stack{
    std::vector<auto_encoder> m_aes; ///< all auto encoders
    auto_enc_stack(unsigned int bs  , unsigned int inp1, int n_layers, const int* layer_sizes, bool binary, float noise=0.0f, float lambda=0.0f){
        m_aes.push_back(auto_encoder(bs  ,inp1,layer_sizes[0], binary,noise,lambda));
        // TODO: do not use noise in 1st layer when training 2nd layer
        for(int i=1;i<n_layers;i++){
            op_ptr out = m_aes.back().m_enc;
            m_aes.push_back(auto_encoder(out,layer_sizes[i],true,noise,lambda));
        }
    }
    auto_encoder& get(unsigned int i){ return m_aes[i]; }
    matrix&        input(){return m_aes.front().input();}
    const matrix& output(){return m_aes.front().output();}
};

/** 
 * for supervised optimization of an objective
 */
struct pretrained_mlp{
    op_ptr    m_input, m_loss;
    input_ptr m_targets;
    input_ptr m_weights;
    input_ptr m_bias;
    op_ptr    m_output; ///< classification result
    output_ptr m_out_sink;
    output_ptr m_loss_sink;
    float m_loss_sum, m_class_err;
    unsigned int m_loss_sum_cnt, m_class_err_cnt;

    // a bunch of convenience functions
    Op::value_type& target(){ return m_targets->data(); }
    void acc_loss()     { m_loss_sum += m_loss_sink->cdata()[0]; }
    void acc_class_err(){ 
        cuv::tensor<int,Op::value_type::memory_space_type> a1 ( m_out_sink->cdata().shape(0) );
        cuv::tensor<int,Op::value_type::memory_space_type> a2 ( m_targets->data().shape(0) );
        cuv::reduce_to_col(a1, m_out_sink->cdata(),cuv::RF_ARGMAX);
        cuv::reduce_to_col(a2, m_targets->data(),cuv::RF_ARGMAX);
        m_class_err     += m_out_sink->cdata().shape(0) - cuv::count(a1-a2,0);
        m_class_err_cnt += m_out_sink->cdata().shape(0);
    }
    float perf(){ return m_class_err/m_class_err_cnt; }
    void reset_loss(){m_loss_sum = m_class_err = m_class_err_cnt = m_loss_sum_cnt = 0; }
    void print_loss(unsigned int epoch){ 
        std::cout <<"MLP: "<< epoch;
        if(m_loss_sum_cnt && m_class_err)
            std::cout << " loss: "<<m_loss_sum/m_loss_sum_cnt << ", ";
        if(m_class_err)
            std::cout << " cerr: "<<m_class_err/m_class_err_cnt<<std::endl;
    }

    pretrained_mlp(op_ptr inputs, unsigned int outputs, bool softmax)
        :m_input(inputs)
    {
        m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
        unsigned int bs   = inputs->result()->shape[0];
        unsigned int inp1 = inputs->result()->shape[1];
        m_weights.reset(new Input(cuv::extents[inp1][outputs],"mlp_weights"));
        m_bias.reset   (new Input(cuv::extents[outputs],      "mlp_bias"));
        m_targets.reset(new Input(cuv::extents[bs][outputs], "mlp_target"));
        m_output = mat_plus_vec( prod( m_input, m_weights) ,m_bias,1);

        // create a sink for outputs so we can determine classification error
        m_out_sink = output(m_output); 
        if(softmax) // multinomial logistic regression
            m_loss = mean(multinomial_logistic_loss(m_output, m_targets,1));
        else        // mean squared error
            m_loss = mean(pow(axpby(-1.f,m_targets,m_output),2.f));
        m_loss_sink = output(m_loss);

        // initialize weights and biases
        float diff = 4.f*std::sqrt(6.f/(inp1+outputs));
        cuv::fill_rnd_uniform(m_weights->data());
        m_weights->data() *= 2*diff;
        m_weights->data() -=   diff;
        m_bias->data()   = 0.f;
        m_bias->data()   = 0.f;
    }
};

/**
 * contains fit and predict functions for a pretrained MLP
 */
struct pretrained_mlp_trainer{
    boost::shared_ptr<pretrained_mlp> m_mlp; ///< the mlp to be trained
    boost::shared_ptr<auto_enc_stack> m_aes; ///< the stacked ae to be pre-trained

    Op::value_type m_current_data;  ///< should contain the dataset we're working on
    Op::value_type m_current_labels; ///< should contain the labels of the dataset we're working on
    Op::value_type m_current_vdata;  ///< should contain the validation dataset we're working on
    Op::value_type m_current_vlabels; ///< should contain the validation labels of the dataset we're working on
    unsigned int m_bs;
    bool m_in_validation_mode; 

    /** 
     * constructor
     * @param mlp the mlp where params should be learned
     */
    pretrained_mlp_trainer(auto_enc_stack* aes, pretrained_mlp* mlp)
        :m_mlp(mlp), m_aes(aes), m_bs(64), m_in_validation_mode(false){}
    void before_validation_epoch(){ m_in_validation_mode = true; }
    void after_validation_epoch(){ m_in_validation_mode = false; }

    /**
     * default constructor for convenience
     */
    pretrained_mlp_trainer(){}

    /**
     * load a batch in an autoencoder
     * @param ae    the autoencoder
     * @param data  the source dataset
     * @param bs    the size of one batch
     * @param batch the number of the requested batch
     */
    void load_batch_ae( 
            auto_encoder* ae, unsigned int batch){
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
            auto_encoder* ae, pretrained_mlp* mlp, unsigned int batch){
        Op::value_type& data = m_in_validation_mode ? m_current_vdata : m_current_data;
        Op::value_type& labl = m_in_validation_mode ? m_current_vlabels : m_current_labels;
        ae->input()   = data[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
        mlp->target() = labl[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
    }

    /**
     * returns classification error on current dataset
     */
    float predict(){
        // "learning" with learnrate 0 and no weight updates
        std::vector<Op*> params;
        gradient_descent gd(m_mlp->m_out_sink,0,params,0.0f,0.0f);
        gd.before_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
        gd.after_epoch.connect(boost::bind(&pretrained_mlp::print_loss, m_mlp.get(),_1));
        gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_mlp,this,&m_aes->get(0),m_mlp.get(),_2));
        gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
        gd.current_batch_num.connect(boost::bind(&pretrained_mlp_trainer::current_batch_num,this));
        gd.minibatch_learning(1,0,0); // 1 epoch
        return m_mlp->m_class_err/m_mlp->m_class_err_cnt;
    }

    /**
     * train the given auto_encoder stack and the mlp
     */
    void fit(){
        ////////////////////////////////////////////////////////////
        //             un-supervised pre-training
        ////////////////////////////////////////////////////////////
        for(unsigned int l=0;l<m_aes->m_aes.size();l++){
            std::vector<Op*> params;
            params += m_aes->get(l).m_weights.get(), m_aes->get(l).m_bias_y.get(), m_aes->get(l).m_bias_h.get();

            gradient_descent gd(m_aes->get(l).m_loss,0,params,0.1f,0.00000f);
            gd.before_epoch.connect(boost::bind(&auto_encoder::reset_loss, &m_aes->get(l)));
            gd.after_epoch.connect(boost::bind(&auto_encoder::print_loss, &m_aes->get(l), _1));
            gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_ae,this,&m_aes->get(0),_2));
            gd.after_batch.connect(boost::bind(&auto_encoder::acc_loss,&m_aes->get(l)));
            gd.current_batch_num.connect(boost::bind(&pretrained_mlp_trainer::current_batch_num,this));
            
            //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails){
            gd.setup_early_stopping(boost::bind(&auto_encoder::perf,&m_aes->get(l)), 5, 0.01f, 2);
            gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp_trainer::before_validation_epoch,this));
            gd.after_validation_epoch.connect( boost::bind(&pretrained_mlp_trainer::after_validation_epoch,this));

            gd.minibatch_learning(10000);
        }
        ////////////////////////////////////////////////////////////
        //                 supervised training
        ////////////////////////////////////////////////////////////
        std::vector<Op*> params;
        for(unsigned int l=0;l<m_aes->m_aes.size();l++) // derive w.r.t. /all/ parameters
            params += m_aes->get(l).m_weights.get(), m_aes->get(l).m_bias_y.get(), m_aes->get(l).m_bias_h.get();
        params += m_mlp->m_weights.get(), m_mlp->m_bias.get();

        gradient_descent gd(m_mlp->m_loss,0,params,0.1f,0.00000f);
        gd.before_epoch.connect(boost::bind(&pretrained_mlp::reset_loss,m_mlp.get()));
        gd.after_epoch.connect(boost::bind(&pretrained_mlp::print_loss,m_mlp.get(), _1));
        gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_mlp,this,&m_aes->get(0),m_mlp.get(),_2));
        gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_loss,m_mlp.get()));
        gd.after_batch.connect(boost::bind(&pretrained_mlp::acc_class_err,m_mlp.get()));
        gd.current_batch_num.connect(boost::bind(&pretrained_mlp_trainer::current_batch_num,this));

        //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails){
        gd.setup_early_stopping(boost::bind(&pretrained_mlp::perf,m_mlp.get()), 5, 0.01f, 2);
        gd.before_validation_epoch.connect(boost::bind(&pretrained_mlp_trainer::before_validation_epoch,this));
        gd.after_validation_epoch.connect( boost::bind(&pretrained_mlp_trainer::after_validation_epoch,this));

        gd.minibatch_learning(10000);
    }
    unsigned int current_batch_num(){ 
        return  m_in_validation_mode ? 
        m_current_vdata.shape(0)/m_bs :
        m_current_data.shape(0)/m_bs; 
    }
};


void prepare_ds(pretrained_mlp_trainer* pmt, splitter* splits, unsigned int split, crossvalidation<pretrained_mlp_trainer>::cv_mode cm){
    std::cout << "switching to split "<<split<<", cm:"<<cm<<std::endl;
    cuv::tensor<float,cuv::host_memory_space> data, vdata;
    cuv::tensor<int,  cuv::host_memory_space> labels, vlabels;
    dataset ds = (*splits)[split];
    switch(cm){
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
    if(vlabels.ndim()){
        cuv::tensor<float,cuv::host_memory_space> fvlabels(vlabels.shape());
        cuv::convert(fvlabels, vlabels);
        pmt->m_current_vdata   = vdata;
        pmt->m_current_vlabels = fvlabels;
    }
}

class pmlp_cv
: public crossvalidation<pretrained_mlp_trainer>
{
    private:
        dataset& m_ds;
        splitter m_splits;
    public:
        pmlp_cv(dataset& ds, unsigned int splits)
        : crossvalidation<pretrained_mlp_trainer>(splits)
        , m_ds(ds)
        , m_splits(ds, splits){
            this->switch_dataset.connect(boost::bind(prepare_ds,_1,&m_splits,_2,_3));
        }
        void generate_and_test_models(){
            unsigned int fa=16,fb=16,bs=64;
            static const int n_layers      = 2;
            static const int layer_size[]  = {fa*fb, fa*fb};
            auto_enc_stack* aes = new auto_enc_stack(
                    bs,
                    m_ds.train_data.shape(1), 
                    n_layers, layer_size, 
                    m_ds.channels==1, 0.00f, 1.000000f);
            pretrained_mlp* mlp = new pretrained_mlp(
                    aes->get(n_layers-1).m_enc, 10, true);
            pretrained_mlp_trainer pmlpt(aes, mlp);  // takes ownership of aes and mlp
            m_io_service.post(boost::bind(&crossvalidation<pretrained_mlp_trainer>::evaluate, this, boost::ref(pmlpt)));
            evaluate(pmlpt);
        }
};

int main(int argc, char **argv)
{
    cuv::initialize_mersenne_twister_seeds();
    std::cout << "main: on device "<<cuv::getCurrentDevice()<<std::endl;

    mnist_dataset ds_all("/home/local/datasets/MNIST");
    global_min_max_normalize<> normalizer(0,1); // 0,1
    //cifar_dataset ds;
    //zero_mean_unit_variance<> normalizer;
    //amat_dataset ds_all("/home/local/datasets/bengio/mnist.zip","mnist_train.amat", "mnist_test.amat");
    //global_min_max_normalize<> normalizer(0,1); // 0,1
    splitter ds_split(ds_all,3);
    dataset ds  = ds_split[0];
    
    normalizer.fit_transform(ds.train_data);
    normalizer.transform(ds.val_data);

    pmlp_cv trainer(ds, 3);
    float best_result = trainer.run();
    std::cout << "best result: "<<best_result<<std::endl;
    
    return 0;
}

