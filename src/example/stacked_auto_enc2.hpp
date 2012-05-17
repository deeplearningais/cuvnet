#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuv.hpp>

#include <cuvnet/ops.hpp>

namespace cuvnet{
    extern cv::crossvalidation_worker* g_worker;
}

using namespace cuvnet;
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
        virtual op_ptr encoded()=0;
        virtual op_ptr decode(op_ptr&)=0;
        virtual op_ptr loss()=0;
        virtual void reset_weights()=0;
        auto_encoder(bool binary):m_binary(binary){}
        bool binary()const{return m_binary;}

        op_ptr reconstruction_loss(op_ptr& input, op_ptr& decode){
            if(!m_binary)  // squared loss
                return mean( pow( axpby(input, -1.f, decode), 2.f));
            else         // cross-entropy
            {
                return mean( sum(neg_log_cross_entropy_of_logistic(input,decode),1));
            }
        }

        virtual void acc_loss()=0;
        unsigned int    avg_epochs()  { 
            if(acc::count(s_epochs)==0) 
                return 0;
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
    public:
        op_ptr       m_input;
        input_ptr    m_weights,m_bias_h,m_bias_y;
        sink_ptr   m_reg_sink, m_rec_sink,m_loss_sink, m_dec_sink;
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

        std::vector<Op*>   supervised_params(){ 
            using namespace boost::assign;
            std::vector<Op*> tmp; tmp += m_weights.get(), m_bias_h.get(); 
            return tmp; 
        };
        std::vector<Op*> unsupervised_params(){
            using namespace boost::assign;
            std::vector<Op*> tmp; tmp += m_weights.get(), m_bias_h.get(), m_bias_y.get(); 
            return tmp; 
        };

        matrix&       input() {
            return boost::dynamic_pointer_cast<Input>(m_input)->data();
        }
        op_ptr encoded() {
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
        auto_encoder_1l(unsigned int layer, op_ptr& inputs, unsigned int hl, bool binary, float noise=0.0f, float wd=0.f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(inputs)
             ,m_bias_h(new Input(cuv::extents[hl],       "ae_bias_h"+ boost::lexical_cast<std::string>(layer))) {
                 m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
                 unsigned int bs   = inputs->result()->shape[0];
                 unsigned int inp1 = inputs->result()->shape[1];
                 m_weights.reset(new Input(cuv::extents[inp1][hl],"ae_weights" + boost::lexical_cast<std::string>(layer)));
                 m_bias_y.reset(new Input(cuv::extents[inp1],     "ae_bias_y"  + boost::lexical_cast<std::string>(layer)));
                 init(bs  ,inp1,hl,binary,noise,wd,lambda);
             }

        /** this constructor is used for the outermost autoencoder in a stack
         * @param bs   batch size
         * @param inp1 number of variables in one pattern
         * @param hl   size of encoded representation
         * @param binary if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_encoder_1l(unsigned int bs  , unsigned int inp1, unsigned int hl, bool binary, float noise=0.0f, float wd=0.f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(new Input(cuv::extents[bs  ][inp1],"ae_input"))
             ,m_weights(new Input(cuv::extents[inp1][hl],"ae_weights"))
             ,m_bias_h(new Input(cuv::extents[hl],       "ae_bias_h"))
             ,m_bias_y(new Input(cuv::extents[inp1],     "ae_bias_y")) {
                 init(bs  ,inp1,hl,binary,noise,wd, lambda);
             }

        /** 
         * Initializes weights and biases
         * 
         */    
        void reset_weights() {
            // initialize weights and biases
            float wnorm = m_weights->data().shape(0)
                +         m_weights->data().shape(1);
            float diff = std::sqrt(6.f/wnorm);
            cuv::fill_rnd_uniform(m_weights->data());
            m_weights->data() *= diff*2.f;
            m_weights->data() -= diff;
            m_bias_h->data()   = 0.f;
            m_bias_y->data()   = 0.f;
        }

    private:
        /**
         * initializes the functions in the AE  according to params given in the
         * constructor
         */
        void init(unsigned int bs  , unsigned int inp1, unsigned int hl, bool binary, float noise, float wd, float lambda) {
            Op::op_ptr corrupt               = m_input;
            if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
            if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);
            if(binary) corrupt = corrupt + -0.5f; // [-.5,.5]
            m_enc    = tanh(mat_plus_vec(
                        prod( corrupt, m_weights)
                        ,m_bias_h,1));
            m_decode = decode(m_enc);
            m_dec_sink = sink("decoded", m_decode);

            m_rec_loss = reconstruction_loss(m_input,m_decode);

            if(wd>0.f){
                m_contractive_loss = mean(pow(m_weights,2.f));
                m_loss = axpby(m_rec_loss, wd, m_contractive_loss);
                m_reg_sink = sink("weight decay loss", m_contractive_loss);
                m_rec_sink = sink("reconstruction loss", m_rec_loss);
            }
            else if(lambda>0.f) { // contractive AE
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
    public:
        op_ptr       m_input;
        input_ptr    m_weights1,m_weights2;
        input_ptr    m_bias_h1a, m_bias_h1b, m_bias_h2, m_bias_y;
        sink_ptr     m_loss_sink, m_reg_sink, m_rec_sink, m_dec_sink;
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
            using namespace boost::assign;
            std::vector<Op*> tmp; 
            tmp += m_weights1.get(), m_weights2.get(), m_bias_h1a.get(), m_bias_h2.get(); 
            return tmp; };
        std::vector<Op*> unsupervised_params(){ 
            using namespace boost::assign;
            std::vector<Op*> tmp; 
            tmp += m_weights1.get(), m_weights2.get(), m_bias_h1a.get(), m_bias_h1b.get(), m_bias_h2.get(), m_bias_y.get(); 
            return tmp; };

        op_ptr decode(op_ptr& encoded){
            op_ptr h1b = tanh(mat_plus_vec( prod( encoded, m_weights2, 'n','t') ,m_bias_h1b,1));
            op_ptr y   = mat_plus_vec( prod( h1b, m_weights1, 'n', 't'), m_bias_y, 1);
            return y;
        }

        matrix&       input() {
            return boost::dynamic_pointer_cast<Input>(m_input)->data();
        }
        op_ptr encoded() {
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
        auto_encoder_2l(unsigned int layer, op_ptr& inputs, unsigned int hl1, unsigned int hl2, bool binary, float noise=0.0f, float wd0=0.0f, float wd1=0.0f, float lambda=0.0f)
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
                 init(bs  ,inp1,hl1,hl2,binary,noise,wd0,wd1,lambda);
             }

        /** this constructor is used for the outermost autoencoder in a stack
         * @param bs   batch size
         * @param inp1 number of variables in one pattern
         * @param hl   size of encoded representation
         * @param binary if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_encoder_2l(unsigned int bs  , unsigned int inp1, unsigned int hl1, unsigned int hl2, bool binary, float noise=0.0f, float wd0=0.0f, float wd1=0.0f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(new Input(cuv::extents[bs  ][inp1],"ae_input"))
             ,m_weights1(new Input(cuv::extents[inp1][hl1],"ae_weights1"))
             ,m_weights2(new Input(cuv::extents[hl1][hl2],"ae_weights2"))
             ,m_bias_h1a(new Input(cuv::extents[hl1],       "ae_bias_h1a"))
             ,m_bias_h1b(new Input(cuv::extents[hl1],       "ae_bias_h1b"))
             ,m_bias_h2 (new Input(cuv::extents[hl2],       "ae_bias_h2"))
             ,m_bias_y(new Input(cuv::extents[inp1],     "ae_bias_y")) {
                 init(bs  ,inp1,hl1,hl2,binary,noise,wd0, wd1, lambda);
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
                float diff = std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_weights1->data());
                m_weights1->data() *= 2*diff;
                m_weights1->data() -=   diff;
            }

            {
                float wnorm = m_weights2->data().shape(0)
                    +         m_weights2->data().shape(1);
                float diff = std::sqrt(6.f/wnorm);
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
        void init(unsigned int bs  , unsigned int inp1, unsigned int hl1, unsigned int hl2, bool binary, float noise, float wd0, float wd1, float lambda) {
            Op::op_ptr corrupt               = m_input;
            if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
            if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);
            if(binary) corrupt = corrupt + -0.5f; // [-.5,.5]

            op_ptr h1  = tanh( mat_plus_vec( prod( corrupt, m_weights1) ,m_bias_h1a,1));
            op_ptr h2  = tanh( mat_plus_vec( prod( h1     , m_weights2) ,m_bias_h2 ,1));
            m_enc      = h2;
            m_decode   = decode(m_enc);
            m_dec_sink = sink("decoded", m_decode);
            m_rec_loss = reconstruction_loss(m_input,m_decode);

            if(wd0 > 0.f || wd1 > 0.f){
                m_contractive_loss = axpby(wd0,mean(pow(m_weights1,2.f)), wd1, mean(pow(m_weights2,2.f)));
                m_loss             = axpby(m_rec_loss, 1.f, m_contractive_loss);
                m_reg_sink         = sink("weight decay loss", m_contractive_loss);
            }
            else if(lambda>0.f) { // contractive AE
                unsigned int num_contr = bs/8;
                for(unsigned int i=0;i<num_contr;i++){
                    op_ptr rs  = row_select(h1,h2); // select same (random) row in h1 and h2
                    op_ptr h1r = result(rs,0);
                    op_ptr h2r = result(rs,1);

                    op_ptr h1_ = h1r*(1.f-h1r);
                    op_ptr h2_ = h2r*(1.f-h2r);

                    op_ptr tmp = sum( sum(pow(prod(mat_times_vec(m_weights1,h1_,1), m_weights2),2.f),0)*pow(h2_,2.f));
                    if(!m_contractive_loss) 
                        m_contractive_loss = tmp;
                    else
                        m_contractive_loss = tmp + m_contractive_loss;

                }
                //op_ptr J      = mat_times_vec(prod(mat_times_vec(m_weights1,h2_,1), m_weights2),h1_,0);
                //m_contractive_loss = sum( pow(J, 2.f) );
                m_loss        = axpby(m_rec_loss, lambda/num_contr, m_contractive_loss);
                m_reg_sink    = sink("contractive loss", m_contractive_loss);
            } else{
                m_loss        = m_rec_loss; // no change
            }
            m_rec_sink    = sink("reconstruction loss", m_rec_loss);
            m_loss_sink   = sink("total loss", m_loss);
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
        op_ptr m_enc;
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
         * @param noise    if noise>0, add noise to the input
         * @param wd       if wd>0, add weight decay
         * @param lambda   if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_enc_stack(unsigned int bs, unsigned int inp1, int n_layers, const int* layer_sizes, bool binary, float* noise, float* wd, float* lambdas, std::vector<bool> twolayer) 
        {
            int i = 0;
            if(twolayer[i]) {
                m_aes.push_back(new auto_encoder_2l(bs  ,inp1,layer_sizes[0],layer_sizes[1], binary,noise[0],wd[0], wd[1], lambdas[0]));
                i+=2;
            }
            else{
                m_aes.push_back(new auto_encoder_1l(bs  ,inp1,layer_sizes[0], binary,noise[0],wd[0], lambdas[0]));
                i+=1;
            }
            // TODO: do not use noise in 1st layer when training 2nd layer
            for(; i<n_layers;) {
                op_ptr out = m_aes.back()->encoded();
                if(twolayer[i]){
                    m_aes.push_back(new auto_encoder_2l(i,out,layer_sizes[i],layer_sizes[i+1],true,noise[i],wd[i], wd[i+1], lambdas[i]));
                    i+=2;
                }
                else{
                    m_aes.push_back(new auto_encoder_1l(i,out,layer_sizes[i],true,noise[i],wd[i],lambdas[i]));
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
        op_ptr encoded() {
            return m_aes.back()->encoded();
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
                m_enc = m_aes.back()->encoded(); // this is the result of the /encoder/
                for(int i=m_aes.size()-1; i>=0; i--) {
                    m_enc = m_aes[i]->decode(m_enc);
                    if(i != 0 ){
                        // we avoided calling logistic in the decoder and let loss deal with it
                        // ----> do that now!
                        // for layer zero, this is again dealt with in the loss!
                        m_enc = logistic(m_enc);    
                    }
                }
                m_combined_loss = m_aes[0]->reconstruction_loss(m_aes[0]->input_op(), m_enc);
                m_loss_sink       = sink("AES combined loss", m_combined_loss);
                m_out_sink        = sink("AES encoded", m_enc);
            }
            return m_combined_loss;
        }
        void log_loss(unsigned int epoch) {
            g_worker->log(BSON("who"<<"AES"<<"epoch"<<epoch<<"perf"<<acc::mean(s_combined_loss)));
            g_worker->checkpoint();
        }
        void acc_loss() {
            s_combined_loss((float)m_loss_sink->cdata()[0]);

            if(m_aes[0]->binary()){
                matrix out(m_out_sink->cdata().shape());
                matrix& in = boost::dynamic_pointer_cast<Input>(m_aes[0]->input_op())->data();
                cuv::apply_binary_functor(out,m_out_sink->cdata(),in,cuv::BF_SUBTRACT);
                cuv::apply_scalar_functor(out,cuv::SF_ABS);

                cuv::apply_scalar_functor(out,out,cuv::SF_LT,0.5f); // out <- abs(true-predicted)<0.5

                cuv::tensor<float, matrix::memory_space_type> col(out.shape(0)); // sum for each batch element
                cuv::reduce_to_col(col,out);
                s_class_err( 1.f - (float) cuv::count(col,0.f)/(float)out.shape(0) ); // average classification acc in batch
            }
        }
        float perf() {
            if(m_aes[0]->binary()){
                // classification loss(!)
                return acc::mean(s_class_err);
            }
            return acc::mean(s_combined_loss);
        }
        void reset_loss() {
            s_combined_loss = acc_t();
            s_class_err     = acc_t();
        }
        unsigned int    avg_epochs()  { 
            if(acc::count(s_epochs)==0) 
                return 0;
            return acc::mean(s_epochs); 
        }
};
