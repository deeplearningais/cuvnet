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

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/derivative_test.hpp>
#include <cuvnet/ops.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <tools/visualization.hpp>
#include <tools/preprocess.hpp>
#include <tools/gradient_descent.hpp>
#include <mongo/bson/bson.h>

#include <cuvnet/op_io.hpp>
#include <tools/crossvalid.hpp>
#include <tools/learner.hpp>
#include <cuv.hpp>

namespace cuvnet{
    extern cv::crossvalidation_worker* g_worker;
}

using namespace cuvnet;
using namespace boost::assign;
namespace acc=boost::accumulators;

typedef boost::shared_ptr<Input>  input_ptr;
typedef boost::shared_ptr<Sink> sink_ptr;
typedef boost::shared_ptr<Op>     op_ptr;

typedef
acc::accumulator_set<double,
    acc::stats<acc::tag::mean, acc::tag::variance(acc::lazy) > > acc_t;

class auto_encoder {
    protected:
        acc_t s_rec_loss; ///< reconstruction
        acc_t s_reg_loss; ///< regularization
        acc_t s_total_loss;

        bool m_binary; ///< if true, use logistic loss

    public:
        acc_t s_epochs;   ///< how many epochs this was trained for
        virtual std::vector<Op*>   supervised_params()=0;
        virtual std::vector<Op*> unsupervised_params()=0;
        virtual matrix& input()=0;
        virtual op_ptr& input_op()=0;
        virtual op_ptr output()=0;
        virtual op_ptr decode(op_ptr&)=0;
        virtual op_ptr loss()=0;
        virtual void reset_weights()=0;
        auto_encoder(bool binary):m_binary(binary){}

        op_ptr reconstruction_loss(op_ptr& input, op_ptr& decode){
            if(!m_binary)  // squared loss
                return mean( pow( axpby(input, -1.f, decode), 2.f));
            else         // cross-entropy
                return mean( sum(neg_log_cross_entropy_of_logistic(input,decode),1));
        }

        virtual void acc_loss()=0;
        unsigned int    avg_epochs()  { 
            if(acc::count(s_epochs)==0) 
                throw std::runtime_error("cannot calculate average epochs!");
            return acc::mean(s_epochs); 
        }
        void reset_loss() {
            s_rec_loss = acc_t();
            s_reg_loss = acc_t();
            s_total_loss = acc_t();
        }
        virtual
        void log_loss(unsigned int epoch) {
            g_worker->log(BSON("who"<<"AE"<<"epoch"<<epoch<<"perf"<<acc::mean(s_total_loss)<<"reg"<<acc::mean(s_reg_loss)<<"rec"<<acc::mean(s_rec_loss)));
            g_worker->checkpoint();
        }
        float perf() {
            return acc::mean(s_total_loss);
        }
};

/**
 * one layer of a stacked auto encoder
 */
class auto_encoder_1l : public auto_encoder{
    private:
        op_ptr       m_input;
        input_ptr    m_weights,m_bias_h,m_bias_y;
        sink_ptr   m_reg_sink, m_rec_sink,m_loss_sink;
        op_ptr       m_decode, m_enc;
        op_ptr       m_loss, m_rec_loss, m_contractive_loss;
    public:
        op_ptr&         input_op(){ return m_input; }
        input_ptr       weights() { return m_weights; }
        input_ptr       bias_h()  { return m_bias_h; }
        input_ptr       bias_y()  { return m_bias_y; }
        op_ptr          loss()    { return m_loss; }
        op_ptr          decoded() { return m_decode; }
        op_ptr          decode(op_ptr& encoded){
            return mat_plus_vec(
                    prod( encoded, m_weights, 'n','t')
                    ,m_bias_y,1);
        }

        std::vector<Op*>   supervised_params(){ std::vector<Op*> tmp; tmp += m_weights.get(), m_bias_h.get(); return tmp; };
        std::vector<Op*> unsupervised_params(){ std::vector<Op*> tmp; tmp += m_weights.get(), m_bias_h.get(), m_bias_y.get(); return tmp; };

        matrix&       input() {
            return boost::dynamic_pointer_cast<Input>(m_input)->data();
        }
        op_ptr output() {
            return m_enc;
        }
        void acc_loss() {
            s_total_loss((float)m_loss_sink->cdata()[0]);
            if(m_rec_sink)
                s_rec_loss((float)m_rec_sink->cdata()[0]);
            if(m_reg_sink)
                s_reg_loss((float)m_reg_sink->cdata()[0]);
        }
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
        auto_encoder_1l(unsigned int layer, op_ptr& inputs, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(inputs)
             ,m_bias_h(new Input(cuv::extents[hl],       "ae_bias_h"+ boost::lexical_cast<std::string>(layer))) {
                 m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
                 unsigned int bs   = inputs->result()->shape[0];
                 unsigned int inp1 = inputs->result()->shape[1];
                 m_weights.reset(new Input(cuv::extents[inp1][hl],"ae_weights" + boost::lexical_cast<std::string>(layer)));
                 m_bias_y.reset(new Input(cuv::extents[inp1],     "ae_bias_y"  + boost::lexical_cast<std::string>(layer)));
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
        auto_encoder_1l(unsigned int bs  , unsigned int inp1, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(new Input(cuv::extents[bs  ][inp1],"ae_input"))
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
            m_decode = decode(m_enc);

            m_rec_loss = reconstruction_loss(m_input,m_decode);

            if(lambda>0.f) { // contractive AE
                m_contractive_loss =
                    sum(sum(pow(m_enc*(1.f-m_enc),2.f),0)
                            * sum(pow(m_weights,2.f),0));
                m_loss        = axpby(m_rec_loss, lambda/(float)bs  , m_contractive_loss);
                m_reg_sink    = sink("contractive loss",m_contractive_loss);
                m_rec_sink    = sink("reconstruction loss", m_rec_loss);
            } else{
                m_loss        =      m_rec_loss; // no change
                m_rec_sink    = sink("reconstruction loss", m_rec_loss);
            }
            m_loss_sink       = sink("total loss", m_loss);
            reset_weights();
        }
};

/**
 * a two-layer auto-encoder
 */
class auto_encoder_2l : public auto_encoder{
    private:
        op_ptr       m_input;
        input_ptr    m_weights1,m_weights2;
        input_ptr    m_bias_h1a, m_bias_h1b, m_bias_h2, m_bias_y;
        sink_ptr     m_loss_sink, m_reg_sink, m_rec_sink;
        op_ptr       m_decode, m_enc;
        op_ptr       m_loss, m_rec_loss, m_contractive_loss;

        float        m_expected_size[2];
    public:
        op_ptr&         input_op(){ return m_input; }
        input_ptr       weights1() { return m_weights1; }
        input_ptr       weights2() { return m_weights2; }
        input_ptr       bias_h1a()  { return m_bias_h1a; }
        input_ptr       bias_h1b()  { return m_bias_h1b; }
        input_ptr       bias_h2()  { return m_bias_h2; }
        input_ptr       bias_y()  { return m_bias_y; }
        op_ptr          loss()    { return m_loss; }
        op_ptr          decoded() { return m_decode; }

        std::vector<Op*>   supervised_params(){ 
            std::vector<Op*> tmp; 
            tmp += m_weights1.get(), m_weights2.get(), m_bias_h1a.get(), m_bias_h2.get(); 
            return tmp; };
        std::vector<Op*> unsupervised_params(){ 
            std::vector<Op*> tmp; 
            tmp += m_weights1.get(), m_weights2.get(), m_bias_h1a.get(), m_bias_h1b.get(), m_bias_h2.get(), m_bias_y.get(); 
            return tmp; };

        op_ptr decode(op_ptr& encoded){
            op_ptr h1b = logistic(mat_plus_vec( prod( encoded, m_weights2, 'n','t') ,m_bias_h1b,1));
            op_ptr y   = mat_plus_vec( prod( h1b, m_weights1, 'n', 't'), m_bias_y, 1);
            return y;
        }

        matrix&       input() {
            return boost::dynamic_pointer_cast<Input>(m_input)->data();
        }
        op_ptr output() {
            return m_enc;
        }
        void acc_loss() {
            s_total_loss((float)m_loss_sink->cdata()[0]);
            if(m_rec_sink)
                s_rec_loss((float)m_rec_sink->cdata()[0]);
            if(m_reg_sink)
                s_reg_loss((float)m_reg_sink->cdata()[0]);
            // TODO: only normalize columns when NOT in validation mode! (why, they should not differ that much in that case...)
            //normalize_columns(m_weights1->data(), m_expected_size[0]);
            //normalize_columns(m_weights2->data(), m_expected_size[1]); // hmm... we have to leave /some/ freedom in the network???
        }
        void normalize_columns(matrix& w, float& expected_size){
            matrix r(w.shape(1));
            cuv::reduce_to_row(r, w, cuv::RF_ADD_SQUARED);
            cuv::apply_scalar_functor(r, cuv::SF_SQRT);
            float f = cuv::mean(r);
            if(expected_size < 0)
                expected_size = f;
            else
                expected_size += 0.99f * (f - expected_size);
            r /= std::max(0.0001f, expected_size);
            cuv::apply_scalar_functor(r, cuv::SF_MAX, 0.00001f); // avoid div by 0
            cuv::matrix_divide_row(w, r);
        }
        virtual
        void log_loss(unsigned int epoch) {
            auto_encoder::log_loss(epoch);
            //std::cout << "Expected Size: "<<m_expected_size[0] << ", "<<m_expected_size[1]<<std::endl;
            // TODO: determine number of saturated units in output/hidden layer
            // TODO: determine Jacobian properties (spectrum)
            // TODO: determine Jacobian of models learned in two separate steps
        }

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
        auto_encoder_2l(unsigned int layer, op_ptr& inputs, unsigned int hl1, unsigned int hl2, bool binary, float noise=0.0f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(inputs)
             ,m_bias_h1a(new Input(cuv::extents[hl1],       "ae_bias_h1a"+ boost::lexical_cast<std::string>(layer))) 
             ,m_bias_h1b(new Input(cuv::extents[hl1],       "ae_bias_h1b"+ boost::lexical_cast<std::string>(layer))) 
             ,m_bias_h2 (new Input(cuv::extents[hl2],       "ae_bias_h2" + boost::lexical_cast<std::string>(layer))) 
        {
                 m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
                 unsigned int bs   = inputs->result()->shape[0];
                 unsigned int inp1 = inputs->result()->shape[1];
                 m_weights1.reset(new Input(cuv::extents[inp1][hl1],"ae_weights1" + boost::lexical_cast<std::string>(layer)));
                 m_weights2.reset(new Input(cuv::extents[hl1][hl2], "ae_weights2" + boost::lexical_cast<std::string>(layer)));
                 m_bias_y.reset(new Input(cuv::extents[inp1],       "ae_bias_y"   + boost::lexical_cast<std::string>(layer)));
                 init(bs  ,inp1,hl1,hl2,binary,noise,lambda);
             }

        /** this constructor is used for the outermost autoencoder in a stack
         * @param bs   batch size
         * @param inp1 number of variables in one pattern
         * @param hl   size of encoded representation
         * @param binary if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_encoder_2l(unsigned int bs  , unsigned int inp1, unsigned int hl1, unsigned int hl2, bool binary, float noise=0.0f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(new Input(cuv::extents[bs  ][inp1],"ae_input"))
             ,m_weights1(new Input(cuv::extents[inp1][hl1],"ae_weights1"))
             ,m_weights2(new Input(cuv::extents[hl1][hl2],"ae_weights2"))
             ,m_bias_h1a(new Input(cuv::extents[hl1],       "ae_bias_h1a"))
             ,m_bias_h1b(new Input(cuv::extents[hl1],       "ae_bias_h1b"))
             ,m_bias_h2 (new Input(cuv::extents[hl2],       "ae_bias_h2"))
             ,m_bias_y(new Input(cuv::extents[inp1],     "ae_bias_y")) {
                 init(bs  ,inp1,hl1,hl2,binary,noise,lambda);
             }

        /** 
         * Initializes weights and biases
         * 
         */    
        void reset_weights() {
            // initialize weights and biases
            {
                float wnorm = m_weights1->data().shape(0)
                    +         m_weights1->data().shape(1);
                float diff = 4.f*std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_weights1->data());
                m_weights1->data() *= 2*diff;
                m_weights1->data() -=   diff;
            }

            {
                float wnorm = m_weights2->data().shape(0)
                    +         m_weights2->data().shape(1);
                float diff = 4.f*std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_weights2->data());
                m_weights2->data() *= 2*diff;
                m_weights2->data() -=   diff;
            }

            m_expected_size[0] = -1;
            m_expected_size[1] = -1;
            //normalize_columns(m_weights1->data(), m_expected_size[0]);
            //normalize_columns(m_weights2->data(), m_expected_size[1]);

            m_bias_h1a->data()   = 0.f;
            m_bias_h1b->data()   = 0.f;
            m_bias_h2 ->data()   = 0.f;
            m_bias_y->data()   = 0.f;
        }

    private:
        /**
         * initializes the functions in the AE  according to params given in the
         * constructor
         */
        void init(unsigned int bs  , unsigned int inp1, unsigned int hl1, unsigned int hl2, bool binary, float noise, float lambda) {
            Op::op_ptr corrupt               = m_input;
            if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
            if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);

            op_ptr h1  = logistic( mat_plus_vec( prod( corrupt, m_weights1) ,m_bias_h1a,1));
            op_ptr h2  = logistic( mat_plus_vec( prod( h1     , m_weights2) ,m_bias_h2 ,1));
            m_enc      = h2;
            m_decode   = decode(m_enc);
            m_rec_loss = reconstruction_loss(m_input,m_decode);

            op_ptr rs = row_select(h1,h2); // select same (random) row in h1 and h2
            op_ptr h1r = result(rs,0);
            op_ptr h2r = result(rs,1);
            
            op_ptr h1_ = h1r*(1.f-h1r);
            op_ptr h2_ = h2r*(1.f-h2r);

            if(lambda>0.f) { // contractive AE
                m_contractive_loss = sum( sum(pow(prod(mat_times_vec(m_weights1,h1_,1), m_weights2),2.f),0)*pow(h2_,2.f));
                //op_ptr J      = mat_times_vec(prod(mat_times_vec(m_weights1,h2_,1), m_weights2),h1_,0);
                //m_contractive_loss = sum( pow(J, 2.f) );
                m_loss        = axpby(m_rec_loss, lambda, m_contractive_loss);
                m_rec_sink    = sink("reconstruction loss", m_rec_loss);
                m_reg_sink    = sink("contractive loss", m_contractive_loss);
            } else
                m_loss        = m_rec_loss; // no change
            m_loss_sink       = sink("total loss", m_loss);
            reset_weights();
        }
};

/**
 * a stack of multiple `auto_encoder's.
 */
struct auto_enc_stack {
    private:
        std::vector<auto_encoder*> m_aes; ///< all auto encoders
        op_ptr m_combined_loss; ///< for finetuning reconstruction of the complete autoencoder
        op_ptr m_output;
        sink_ptr     m_loss_sink, m_out_sink; ///< sink for combined loss
        acc_t  s_combined_loss; ///< statistics for combined loss
        acc_t  s_class_err; ///< classification error
    public:
        acc_t  s_epochs;
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
        auto_enc_stack(unsigned int bs, unsigned int inp1, int n_layers, const int* layer_sizes, bool binary, float* noise, float* lambdas, std::vector<bool> twolayer) 
        {
            int i = 0;
            if(twolayer[i]) {
                m_aes.push_back(new auto_encoder_2l(bs  ,inp1,layer_sizes[0],layer_sizes[1], binary,noise[0],lambdas[0]));
                i+=2;
            }
            else{
                m_aes.push_back(new auto_encoder_1l(bs  ,inp1,layer_sizes[0], binary,noise[0],lambdas[0]));
                i+=1;
            }
            // TODO: do not use noise in 1st layer when training 2nd layer
            for(; i<n_layers;) {
                op_ptr out = m_aes.back()->output();
                if(twolayer[i]){
                    m_aes.push_back(new auto_encoder_2l(i,out,layer_sizes[i],layer_sizes[i+1],true,noise[i],lambdas[i]));
                    i+=2;
                }
                else{
                    m_aes.push_back(new auto_encoder_1l(i,out,layer_sizes[i],true,noise[i],lambdas[i]));
                    i+=1;
                }
            }
        }
        /**
         * get a specific auto encoder
         * @param i the layer number of the AE
         * @return the i-th AE
         */
        auto_encoder& get(unsigned int i) {
            return *m_aes[i];
        }
        matrix&        input() {
            return m_aes.front()->input();
        }
        op_ptr output() {
            return m_aes.back()->output();
        }
        unsigned int size()const{
            return m_aes.size();
        }
        void reset_weights() {
            for(unsigned int i=0; i<m_aes.size(); i++) {
                m_aes[i]->reset_weights();
            }
        }
        op_ptr combined_rec_loss(){
            if(!m_combined_loss){
                m_output = m_aes.back()->output();
                for(int i=m_aes.size()-1; i>=0; i--) {
                    m_output = m_aes[i]->decode(m_output);
                }
                m_combined_loss = m_aes[0]->reconstruction_loss(m_aes[0]->input_op(), m_output);
                m_loss_sink       = sink("AES combined loss", m_combined_loss);
                m_out_sink        = sink("AES output", m_output);
            }
            return m_combined_loss;
        }
        void log_loss(unsigned int epoch) {
            g_worker->log(BSON("who"<<"AES"<<"epoch"<<epoch<<"perf"<<acc::mean(s_combined_loss)));
            g_worker->checkpoint();
        }
        void acc_loss() {
            s_combined_loss((float)m_loss_sink->cdata()[0]);

            matrix out(m_out_sink->cdata().shape());
            matrix& in = boost::dynamic_pointer_cast<Input>(m_aes[0]->input_op())->data();
            cuv::apply_binary_functor(out,m_out_sink->cdata(),in,cuv::BF_SUBTRACT);
            cuv::apply_scalar_functor(out,cuv::SF_ABS);

            cuv::apply_scalar_functor(out,out,cuv::SF_LT,0.5f); // out <- abs(true-predicted)<0.5

            cuv::tensor<float, matrix::memory_space_type> col(out.shape(0)); // sum for each batch element
            cuv::reduce_to_col(col,out);
            s_class_err( 1.f - (float) cuv::count(col,0.f)/(float)out.shape(0) ); // average classification acc in batch
        }
        float perf() {
            // classification loss(!)
            return acc::mean(s_class_err);
        }
        void reset_loss() {
            s_combined_loss = acc_t();
            s_class_err     = acc_t();
        }
        unsigned int    avg_epochs()  { 
            if(acc::count(s_epochs)==0) 
                throw std::runtime_error("cannot calculate average epochs!");
            return acc::mean(s_epochs); 
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
        float m_loss_sum, m_class_err;
        unsigned int m_loss_sum_cnt, m_class_err_cnt;
    public:
        acc_t    s_epochs;

        unsigned int    avg_epochs()  { return acc::mean(s_epochs); }
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
        void log_loss(unsigned int epoch) {
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
                m_out_sink = sink("MLP output",m_output);
                if(softmax) // multinomial logistic regression
                    m_loss = mean(multinomial_logistic_loss(m_output, m_targets,1));
                else        // mean squared error
                    m_loss = mean(pow(axpby(-1.f,m_targets,m_output),2.f));
                m_loss_sink = sink("MLP loss",m_loss);

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
                new pretrained_mlp(m_aes->output(),10, true)); // TODO: fixed number of 10 classes!!???
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
            }else if(m_unsupervised_finetune){
                std::vector<Op*> params;
                gradient_descent gd(m_aes->combined_rec_loss(),0,params, 0.0f,0.00000f);
                gd.before_epoch.connect(boost::bind(&auto_enc_stack::reset_loss, m_aes.get()));
                gd.after_epoch.connect(boost::bind(&auto_enc_stack::log_loss, m_aes.get(), _1));
                gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_unsupervised,this,_2));
                gd.after_batch.connect(boost::bind(&auto_enc_stack::acc_loss,m_aes.get()));
                gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));
                gd.minibatch_learning(1,0,0); // 1 epoch
                return m_aes->perf();
            }else{
                cuvAssert(false);
            }
            return -2.f;
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
                        gd.setup_early_stopping(boost::bind(&auto_encoder::perf,&m_aes->get(l)), 5, 0.0001f, 2);
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
                    log_params("after_pretrain", params);
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

                float learnrate = std::accumulate(m_aes_lr.begin(),m_aes_lr.end(),0.f)/m_aes_lr.size();
                gradient_descent gd(m_aes->combined_rec_loss(),0,params, learnrate,0.00000f);
                gd.before_epoch.connect(boost::bind(&auto_enc_stack::reset_loss, m_aes.get()));
                gd.after_epoch.connect(boost::bind(&auto_enc_stack::log_loss, m_aes.get(), _1));
                gd.before_batch.connect(boost::bind(&pretrained_mlp_trainer::load_batch_unsupervised,this,_2));
                gd.after_batch.connect(boost::bind(&auto_enc_stack::acc_loss,m_aes.get()));
                gd.current_batch_num.connect(boost::bind(&sdl_t::n_batches,&m_sdl));

                if(m_sdl.can_earlystop()) {
                    // we can only use early stopping when validation data is given
                    //setup_early_stopping(T performance, unsigned int every_nth_epoch, float thresh, unsigned int maxfails)
                    gd.setup_early_stopping(boost::bind(&auto_enc_stack::perf,m_aes.get()), 5, 0.00000001f, 2);
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

                log_params("after_unsup_finetune", params);
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
                log_params("after_sup_finetune", params);
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
            m_aes->input()  = m_sdl.get_data_batch(batch);
            m_mlp->target() = m_sdl.get_label_batch(batch);
        }
        void load_batch_unsupervised(unsigned int batch){
            m_aes->input()  = m_sdl.get_data_batch(batch);
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
    std::cout << "o:"<<n_open<<" f:"<<n_finished<<" a:"<<n_assigned<<std::endl;
    if(n_open<3){
        boost::shared_ptr<crossvalidatable> p(new pretrained_mlp_trainer());
        std::cout <<"generating new sample"<<std::endl;

        //unsigned int n_layers = 1+3*drand48();
        unsigned int n_layers = 2;

        //float mlp_lr  = 0.1;
        //float aes_lr0  = 0.01;
        float mlp_lr  = log_uniform(0.01, 0.2);
        float aes_lr0  = log_uniform(0.01, 0.2);
        std::vector<float> lambda(n_layers);
        std::vector<float> aes_lr(n_layers);
        std::vector<float> noise(n_layers);
        std::vector<int  > size(n_layers);
        std::vector<bool > twolayer(n_layers);

        float lambda0 = uniform(0.0001, 2.0);
        if(drand48()<0.1)
            lambda0 = 0.f;

        for (unsigned int i = 0; i < n_layers; ++i)
        {
            lambda[i] = lambda0;
            aes_lr[i] = aes_lr0;
            noise[i]  = 0.0;
            size[i]   = 
                (i==0 ? 1000 : 500 );
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
            bob << "dataset" << "cifar";
            bob << "bs"      << 16;
            bob << "nsplits" << 1;
            bob << "mlp_lr"  << mlp_lr;

            bob << "pretrain" << (drand48()>0.1f);
            bob << "ufinetune" << false;
            bob << "sfinetune" << true;

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

void generate_and_test_models_ldpc(boost::asio::deadline_timer* dt, boost::asio::io_service* io, cv::crossvalidation_queue* q) {
    size_t n_open     = q->m_hub.get_n_open();
    size_t n_finished = q->m_hub.get_n_ok();
    size_t n_assigned = q->m_hub.get_n_assigned();
    std::cout << "o:"<<n_open<<" f:"<<n_finished<<" a:"<<n_assigned<<std::endl;
    if(n_open<3){
        boost::shared_ptr<crossvalidatable> p(new pretrained_mlp_trainer());
        std::cout <<"generating new sample"<<std::endl;

        //unsigned int n_layers = 1+3*drand48();
        unsigned int n_layers = 2;

        //float mlp_lr  = 0.1;
        //float aes_lr0  = 0.01;
        float mlp_lr  = log_uniform(0.01, 0.2);
        float aes_lr0  = log_uniform(0.01, 0.2);
        std::vector<float> lambda(n_layers);
        std::vector<float> aes_lr(n_layers);
        std::vector<float> noise(n_layers);
        std::vector<int  > size(n_layers);
        std::vector<bool > twolayer(n_layers);

        float lambda0 = log_uniform(0.000001, 2.0);
        if(drand48() < 0.1)
            lambda0 = 0.f;
        for (unsigned int i = 0; i < n_layers; ++i)
        {
            lambda[i] = lambda0;
            aes_lr[i] = aes_lr0;
            noise[i]  = 0.0;
            size[i]   = 
                ((i==0) ? 1000 : 500);// hidden0: 4*message plus message, hidden1: only message
            twolayer[i] = (i<n_layers-1);
        }

        //
        // idx0=0 : twolayer == true     (train two layers at once)
        // idx0=1 : twolayer == false    (greedy pretraining)
        // idx0=2 : only one layer       (single layer)
        for (int idx0 = 0; idx0 < 3; ++idx0)
        {
            mongo::BSONObjBuilder bob;
            bob << "dataset" << "natural";
            bob << "bs"      << 16;
            bob << "nsplits" << 1;
            bob << "mlp_lr"  << mlp_lr;

            bob << "pretrain" << (drand48()>0.1f);
            //bob << "pretrain" << true;
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
    dt->async_wait(boost::bind(generate_and_test_models_ldpc, dt, io, q));
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

    boost::asio::io_service io;
    if(std::string("hub") == argv[1]){
        cv::crossvalidation_queue q("131.220.7.92","test.twolayer_ae_natural");
        std::cout << "Clear database? type `yes'" << std::endl;
        std::string s;
        std::cin >> s;
        if(s=="yes"){
            std::cout << "clearing....." << std::endl;
            q.m_hub.clear_all();
        }
        boost::asio::deadline_timer dt(io, boost::posix_time::seconds(1));
        dt.async_wait(boost::bind(generate_and_test_models_ldpc, &dt, &io, &q));

        q.m_hub.reg(io,1); // checks for timeouts
        io.run();
    }
    if(std::string("client") == argv[1]){
        cuvAssert(argc==3);
        cuv::initCUDA(boost::lexical_cast<int>(argv[2]));
        cuv::initialize_mersenne_twister_seeds(time(NULL));
        cv::crossvalidation_worker w("131.220.7.92","test.twolayer_ae_natural");
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
        dt.async_wait(boost::bind(generate_and_test_models_ldpc, &dt, &io, &q));
        q.m_hub.clear_all();

        q.m_hub.reg(io,1); // checks for timeouts
        w.reg(io,1);

        io.run();
    }

    return 0;
}

