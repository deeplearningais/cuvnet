#include <boost/assign.hpp>
#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuv.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

#include <datasets/natural.hpp>
#include <datasets/cifar.hpp>
#include <datasets/mnist.hpp>
#include <datasets/splitter.hpp>
#include <datasets/randomizer.hpp>

#include <tools/crossvalid.hpp>
#include <tools/preprocess.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/visualization.hpp>
#include <tools/orthonormalization.hpp>
#include <tools/dumper.hpp>
#include <tools/logging.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>


using namespace cuvnet;
namespace acc=boost::accumulators;
namespace ll = boost::lambda;
using boost::make_shared;

typedef boost::shared_ptr<Input>  input_ptr;
typedef boost::shared_ptr<Sink> sink_ptr;
typedef boost::shared_ptr<Op>     op_ptr;

typedef
acc::accumulator_set<double,
    acc::stats<acc::tag::mean, acc::tag::variance(acc::lazy) > > acc_t;

template<class V, class T>
matrix trans(const cuv::tensor<V,T>& m){
    cuv::tensor<V,T> mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}

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
        virtual op_ptr loss()=0;
        virtual void reset_weights()=0;
        auto_encoder(bool binary):m_binary(binary){}
        bool binary()const{return m_binary;}

        op_ptr reconstruction_loss(op_ptr& input, op_ptr& decode){
            if(!m_binary)  // squared loss
                return mean( pow( axpby(input, -1.f, decode), 2.f));
            else         // cross-entropy
            {
                return mean( sum_to_vec(neg_log_cross_entropy_of_logistic(input,decode),0));
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
        void print_loss(unsigned int epoch) {
            std::cout << "epoch " << epoch<< " perf: "<<acc::mean(s_total_loss)<<" reg: "<<acc::mean(s_reg_loss)<<" rec: "<<acc::mean(s_rec_loss) << std::endl;
            //g_worker->log(BSON("who"<<"AE"<<"epoch"<<epoch<<"perf"<<acc::mean(s_total_loss)<<"reg"<<acc::mean(s_reg_loss)<<"rec"<<acc::mean(s_rec_loss)));
            //g_worker->checkpoint();
            static std::ofstream os("log.txt");
            os << acc::mean(s_total_loss) << " ";
            os << acc::mean(s_reg_loss) << " ";
            os << acc::mean(s_rec_loss);
            os << std::endl;
        }
        float perf() {
            return acc::mean(s_total_loss);
        }
};

/**
 * a relational auto-encoder (see memisevic 2011)
 */
class auto_encoder_rel : public auto_encoder{
    public:
        op_ptr       m_input;
        input_ptr    m_weights_x, m_weights_x0, m_weights_x1, m_weights_h;
        input_ptr    m_meanweights, m_meanbias;
        input_ptr    m_bias_xvis; /// bias of the reconstruction
        input_ptr    m_bias_h;    /// bias of the hidden layer
        sink_ptr     m_loss_sink, m_reg_sink, m_rec_sink, m_decode_sink;
        sink_ptr     m_tmp_sink;
        op_ptr       m_decode, m_enc;
        op_ptr       m_loss, m_rec_loss, m_reg_loss;
        float        m_expected_size[2];

    public:
        op_ptr&         input_op(){ return m_input; }
        op_ptr          loss()    { return m_loss; }
        op_ptr          decoded() { return m_decode; }

        std::vector<Op*>   supervised_params(){ 
            using namespace boost::assign;
            std::vector<Op*> tmp; 
            tmp += m_weights_x0.get();
            tmp += m_weights_x1.get();
            //tmp += m_meanweights.get();
            //tmp += m_meanbias.get();
            //tmp += m_weights_h.get();
            //tmp += m_bias_h.get();
            return tmp; 
        };
        std::vector<Op*> unsupervised_params(){ 
            using namespace boost::assign;
            std::vector<Op*> tmp = supervised_params();
            tmp += m_bias_xvis.get();
            return tmp; 
        };

        matrix&  input() {
            return boost::dynamic_pointer_cast<Input>(m_input)->data();
        }
        op_ptr encoded() {
            return m_enc;
        }
        void acc_loss() {
            if(m_loss_sink) s_total_loss((float)m_loss_sink->cdata()[0]);
            if(m_rec_sink)  s_rec_loss  ((float)m_rec_sink->cdata()[0]);
            if(m_reg_sink)  s_reg_loss  ((float)m_reg_sink->cdata()[0]);


            project_weights();
        }
        void project_weights(){
            return;
            //normalize_columns(m_weights_x->data(), m_expected_size[0]);
            //
            // combine m_weights_x0 and m_weights_x1, normalize, take apart again.
            using namespace cuv;
            unsigned int hl = m_weights_x0->data().shape(1);
            cuvnet::matrix tmp(extents[m_weights_x0->data().shape(0)][hl*2]);

            tmp[indices[index_range()][index_range(0,hl)   ]] = m_weights_x0->data();
            tmp[indices[index_range()][index_range(hl,2*hl)]] = m_weights_x1->data();
            //std::cout << "sum(tmp) before:" << sum(tmp) << std::endl;
            normalize_columns(tmp, m_expected_size[0]);
            //std::cout << "sum(tmp) after :" << sum(tmp) << std::endl;
            m_weights_x0->data() = tmp[indices[index_range()][index_range(0,hl)   ]].copy();
            m_weights_x1->data() = tmp[indices[index_range()][index_range(hl,2*hl)]].copy();
        }
        void angle_stats(const cuv::tensor<float,cuv::host_memory_space>& m){
            // determine angles between successive feature pairs
            acc_t s_theta;
            std::ofstream os("angles.txt");
            for (int i = 0; i < (int)m.shape(1)-1; i+=2)
            {
                float s = 0, b1=0, b2=0;
                for (int x = 0; x < (int)m.shape(0); ++x)
                {
                    s  += m(x,i)   * m(x,i+1);
                    b1 += m(x,i)   * m(x,i);
                    b2 += m(x,i+1) * m(x,i+1);
                }
                b1 = sqrt(b1);
                b2 = sqrt(b2);
                float theta = acos(std::min(1.f,std::max(-1.f,s/b1/b2)));
                s_theta(theta*180.f/(float)M_PI);
                os << theta<<std::endl;
            }
            std::cout << "acc::mean(s_theta):" << acc::mean(s_theta) << " acc::var(s_theta):" << acc::variance(s_theta) << std::endl;/* cursor */
        }
        void normalize_columns(matrix& w, float& expected_size){
            matrix mean(w.shape(1));
            matrix var (w.shape(1));

            static unsigned int cnt=0;
            //if(cnt++ > 100000 && cnt%100==0)
            bool step2 = cnt++>1;
            bool ortho = false;
            if(ortho && step2) {
                if(cnt%100==0)
                //orthogonalize_pairs(w,true);
                orthogonalize_symmetric(w,true);
            }

            if(false && !step2){
                // subtract filter means
                cuv::reduce_to_row(mean, w, cuv::RF_ADD);
                mean /= (float) w.shape(0);  // mean of each filter
                cuv::apply_scalar_functor(mean, cuv::SF_NEGATE);
                cuv::matrix_plus_row(w,mean);
            }

            // when orthonormalizing, we just need growing for fade-in...
            if(ortho && expected_size > 0.99f){
                static bool stopped = false;
                if(!stopped){
                    stopped = true;
                    std::cout << "stopping length normalization after "<<cnt<<" steps" << std::endl;
                }
                return;
            }

            // enforce small variance (determined by running average)
            cuv::reduce_to_row(var, w, cuv::RF_ADD_SQUARED);
            cuv::apply_scalar_functor(var, cuv::SF_SQRT);
            float f = cuv::mean(var);
            //std::cout << "expected_size:" << expected_size << " f:" << f  << std::endl;
            if(expected_size < 0)
                expected_size = f;
            else
                expected_size += 0.05f * std::max(0.0001f, f - expected_size); // always grow slowly
            expected_size = std::min(1.f, expected_size);
            //std::cout << "expected_size:" << expected_size  << std::endl;

            cuv::apply_scalar_functor(var, cuv::SF_MAX, 0.0001f);
            cuv::apply_scalar_functor(var, cuv::SF_RDIV, expected_size);
            cuv::matrix_times_row(w, var);

            //if(step2)
                //angle_stats(w);

            // add filter mean again
            //cuv::apply_scalar_functor(mean, cuv::SF_NEGATE);
            //cuv::matrix_plus_row(w,mean);
        }
        virtual
        void print_loss(unsigned int epoch) {
            auto_encoder::print_loss(epoch);
            std::cout << "TmpStats  mean: " << cuv::mean(m_tmp_sink->cdata()) << std::endl;
            std::cout << "           var: " << cuv::var(m_tmp_sink->cdata()) << std::endl;
            std::cout << "           min: " << cuv::minimum(m_tmp_sink->cdata()) << std::endl;
            std::cout << "           max: " << cuv::maximum(m_tmp_sink->cdata()) << std::endl;

        }

        /**
         * this constructor gets the \e encoded output of another autoencoder as
         * its input and infers shapes from there.
         *
         * @param layer the number of the layer (used for naming the parameters)
         * @param inputs the "incoming" (=lower level) encoded representation
         * @param hl     size of encoded representation
         * @param factorsize size of the factors the inputs and hiddens are projected to
         * @param binary if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_encoder_rel(unsigned int layer, op_ptr& inputs, unsigned int hl, unsigned int factorsize, bool binary, float noise=0.0f, float lambda=0.0f)
            :auto_encoder(binary),
            m_input(inputs)
        {
                 m_input->visit(determine_shapes_visitor()); // ensure that we have shape information
                 unsigned int bs   = inputs->result()->shape[0];
                 unsigned int inp1 = inputs->result()->shape[1];
                 m_weights_x.reset(new Input(cuv::extents[inp1][factorsize],"ae_wx" + boost::lexical_cast<std::string>(layer)));
                 m_weights_x0.reset(new Input(cuv::extents[inp1][hl],"ae_wx0" + boost::lexical_cast<std::string>(layer)));
                 m_weights_x1.reset(new Input(cuv::extents[inp1][hl],"ae_wx1" + boost::lexical_cast<std::string>(layer)));
                 m_weights_h.reset(new Input(cuv::extents[factorsize][hl],  "ae_wh" + boost::lexical_cast<std::string>(layer)));
                 m_bias_xvis.reset(new Input(cuv::extents[inp1],     "ae_bias_xvis" + boost::lexical_cast<std::string>(layer)));
                 m_bias_h   .reset(new Input(cuv::extents[hl],       "ae_bias_h"    + boost::lexical_cast<std::string>(layer)));
                 m_meanweights.reset(new Input(cuv::extents[inp1][hl/2],  "ae_wmx"));
                 m_meanbias.reset(new Input(cuv::extents[hl/2],  "ae_bmx"));
                 init(bs  ,inp1,hl,factorsize,binary,noise,lambda);
             }

        /** this constructor is used for the outermost autoencoder in a stack
         * @param bs   batch size
         * @param inp1 number of variables in one pattern
         * @param hl   size of encoded representation
         * @param factorsize size of the factors the inputs and hiddens are projected to
         * @param binary if true, logistic function is applied to outputs
         * @param noise  if noise>0, add noise to the input
         * @param lambda if lambda>0, it represents the "contractive" weight of the autoencoder
         */
        auto_encoder_rel(unsigned int bs  , unsigned int inp1, unsigned int hl, unsigned int factorsize, bool binary, float noise=0.0f, float lambda=0.0f)
            :auto_encoder(binary)
            ,m_input(new Input(cuv::extents[bs  ][inp1],"ae_input"))
            ,m_weights_x(new Input(cuv::extents[inp1][factorsize],"ae_wx"))
            ,m_weights_x0(new Input(cuv::extents[inp1][hl],"ae_wx0"))
            ,m_weights_x1(new Input(cuv::extents[inp1][hl],"ae_wx1"))
            ,m_weights_h(new Input(cuv::extents[factorsize][hl],  "ae_wh"))
            ,m_meanweights(new Input(cuv::extents[inp1][hl/2],  "ae_wmx"))
            ,m_meanbias(new Input(cuv::extents[hl/2],  "ae_bmx"))
            ,m_bias_xvis(new Input(cuv::extents[inp1],     "ae_bias_xvis")) 
            ,m_bias_h   (new Input(cuv::extents[hl],     "ae_bias_h")) 
        {
                 init(bs  ,inp1,hl,factorsize,binary,noise,lambda);
             }

        /** 
         * Initializes weights and biases
         * 
         */    
        void reset_weights() {
            // initialize weights and biases
            {
                float wnorm = m_meanweights->data().shape(0)
                    +         m_meanweights->data().shape(1) + 1;
                float diff = 4.f*std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_meanweights->data());
                m_meanweights->data() *= 2*diff;
                m_meanweights->data() -=   diff;
                m_meanbias->data() = 0.f;
            }
            {
                float wnorm = m_weights_h->data().shape(0)
                    +         m_weights_h->data().shape(1) + 1;
                float diff = 4.f*std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_weights_h->data());
                //m_weights_h->data() *= 2*diff;
                //m_weights_h->data() -=   diff;
                m_weights_h->data() -= 3.f;
                cuv::apply_scalar_functor(m_weights_h->data(), cuv::SF_EXP); // from memisevic code, i guess to make h sparse

                if(1){
                    // 1d-topology, as in memisevic code. 
                    matrix& m = m_weights_h->data();
                    m = 0.f;
                    int hls = m.shape(1);
                    int fs  = m.shape(0);
                    float step = std::max(fs,hls) / (float) std::min(fs,hls);
                    cuv::tensor<float,cuv::host_memory_space> kern(2*step+1);
                    for (int i = -step; i < step; ++i)
                    {
                        kern(i+step) = exp(i*i/(step/3.f));
                    }
                    kern /= cuv::sum(kern);
                    if(fs>hls){
                        for(int i=0;i<hls;i++){
                            for(int j=0; j<step; j++){
                                int idx = i*step+j;
                                if(idx >= 0 && idx < fs)
                                    //m(idx, i) = j==0 ? 1.f : -1.f;
                                    m(idx, i) = 1.f;
                            }
                        }
                    }else{
                        for(int i=0;i<fs;i++){
                            for(int j=0; j<step; j++){
                                int idx = i*step+j;
                                if(idx >= 0 && idx < hls)
                                    //m(i,idx) = j == 0 ? 1.f : -1.f;
                                    m(i,idx) = 1.f;
                            }
                        }
                    }
                }else if(0){
                    // 2d-topology
                    matrix& m = m_weights_h->data();
                    m = 0.f;
                    int hls = m.shape(1);
                    int fs  = m.shape(0);

                    int hx = sqrt(hls);
                    int hy = hls/hx;
                    cuvAssert(hx*hy==hls);
                    float stepi = 1;
                    float stepj = 1;
                    for (int i = 0; i < hx; ++i)
                    {
                        for (int j = 0; j < hy; ++j)
                        {
                            for (int si = -stepi; si < stepi; ++si)
                            {
                                for (int sj = -stepj; sj < stepj; ++sj)
                                {
                                    // TODO
                                }
                            }
                        }
                    }
                }
            }

            {
                float wnorm = m_weights_x->data().shape(0)
                    +         m_weights_x->data().shape(1) + 1;
                float diff = 4.f*std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_weights_x->data());
                diff = 0.01f;
                m_weights_x->data() *= 2*diff;
                m_weights_x->data() -=   diff;
            }
            {
                float wnorm = m_weights_x0->data().shape(0)
                    +         m_weights_x0->data().shape(1) + 1;
                float diff = 4.f*std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_weights_x0->data());
                //diff = 0.01f;
                m_weights_x0->data() *= 2*diff;
                m_weights_x0->data() -=   diff;
            }
            {
                float wnorm = m_weights_x1->data().shape(0)
                    +         m_weights_x1->data().shape(1) + 1;
                float diff = 4.f*std::sqrt(6.f/wnorm);
                cuv::fill_rnd_uniform(m_weights_x1->data());
                //diff = 0.01f;
                m_weights_x1->data() *= 2*diff;
                m_weights_x1->data() -=   diff;
            }

            m_expected_size[0] = -1;
            m_expected_size[1] = -1;

            m_bias_xvis->data() = 0.f;
            m_bias_h->data()    = 0.f;

            project_weights(); // normalizes the weights, setting initial expected sizes *cough*
        }

    private:
        /**
         * initializes the functions in the AE  according to params given in the
         * constructor
         */
        void init(unsigned int bs  , unsigned int inp1, unsigned int hl, unsigned int factorsize, bool binary, float noise, float lambda) {
            Op::op_ptr corruptx              = m_input;
            Op::op_ptr corrupty              = m_input;
            if( binary && noise>0.f) corruptx =       zero_out(m_input,noise);
            if(!binary && noise>0.f) corruptx = add_rnd_normal(m_input,noise);
            if( binary && noise>0.f) corrupty =       zero_out(m_input,noise);
            if(!binary && noise>0.f) corrupty = add_rnd_normal(m_input,noise);

            //op_ptr x_ = prod(corruptx, m_weights_x);
            op_ptr x0 = prod(corruptx, m_weights_x0);
            op_ptr x1 = prod(corruptx, m_weights_x1);

            enum functype {FT_TANH, FT_LOGISTIC, FT_NORM, FT_1MEXPMX, FT_ABS} ft = FT_NORM;

            float norm = 2.f;
            bool norm_is_natural = fabs((int)(norm)-norm) < 0.001f;
            bool norm_is_even    = norm_is_natural && (((int)norm)%2 == 0);
            std::cout << "norm:" << norm  << std::endl;
            std::cout << "norm_is_natural:" << norm_is_natural << std::endl;
            std::cout << "norm_is_even:" << norm_is_even << std::endl;

            bool use_same_input_twice = true;
            op_ptr y_, xsq, xsq0,xsq1;
            op_ptr x0_nl = x0, x1_nl = x1;
            if(use_same_input_twice){
                //y_ = x_;
                // apply a saturating non-linearity to responses
                //x0_nl = tanh(x0);
                //x1_nl = tanh(x1);
                if(norm_is_even)     {
                    //xsq = pow(x_, norm);
                    xsq0 = pow(x0_nl, norm);
                    xsq1 = pow(x1_nl, norm);
                }
                else if(norm == 1.f) {
                    //xsq = abs(x_);
                    xsq0 = abs(x0_nl);
                    xsq1 = abs(x1_nl);
                }
                else {
                    //xsq = pow(abs(x_), norm);
                    xsq0 = pow(abs(x0_nl), norm);
                    xsq1 = pow(abs(x1_nl), norm);
                }
            }else{
                cuvAssert(false);
                //y_ = prod(corrupty, m_weights_x);
                //xsq = x_*y_;
            }


            //op_ptr m_enc_lin = mat_plus_vec( prod(xsq, m_weights_h), m_bias_h, 1);
            op_ptr m_enc_lin = xsq0+xsq1;

            switch(ft){
                case FT_TANH: 
                    m_enc     = tanh(m_enc_lin);
                    break;
                case FT_LOGISTIC:
                    m_enc     = logistic(m_enc_lin);
                    break;
                case FT_NORM:
                    m_enc     = pow(m_enc_lin+0.001f, 1.f/norm);
                    break;
                case FT_ABS:
                    m_enc     = abs(m_enc_lin);
                    break;
                case FT_1MEXPMX:
                    m_enc     = 1-exp(-1.f,m_enc_lin);
                    break;
            }
            
            // TODO: memisevic just multiplies by m_enc by x0,x1 here -- why?
            //op_ptr x0rec = x0;
            //op_ptr x1rec = x1;
            op_ptr angle = atan2(x1,x0);
            op_ptr x0rec = m_enc*cos(angle);
            op_ptr x1rec = m_enc*sin(angle);
            m_decode = 
                prod(x0rec, m_weights_x0, 'n','t') +
                prod(x1rec, m_weights_x1, 'n','t') ;
            m_decode = mat_plus_vec(m_decode, m_bias_xvis,1);

            // check whether polar transformation works
            //m_reg_loss = mean(pow(x0-x0rec,2.f) + pow(x1-x1rec,2.f));
            //m_reg_sink = sink("polar-reconstruct", m_reg_loss);

            //op_ptr hp = prod(m_enc, m_weights_h, 'n','t'); // watch out that this is not 1 all the time!
            m_tmp_sink = sink("tmp",m_enc);

            //m_decode      = mat_plus_vec(prod(hp*pow(y_,-1.f),m_weights_x,'n','t'),m_bias_xvis,1);
            //m_decode      = mat_plus_vec(prod(hp*y_,m_weights_x,'n','t'),m_bias_xvis,1);


            bool use_means = false;
            if(use_means){
                m_decode = m_decode + prod(logistic(mat_plus_vec(prod(corruptx, m_meanweights),m_meanbias,1)),m_meanweights,'n','t');
            }

            m_decode_sink = sink("decoded", m_decode);
            m_rec_loss    = reconstruction_loss(m_input,m_decode);

            m_loss        = m_rec_loss; // no change
            m_rec_sink    = sink("reconstruction loss", m_rec_loss);

            //m_loss = axpby(m_loss, 0.f, m_reg_loss); // TODO: remove

            if(lambda>0.f) { // contractive AE
                unsigned int numcontr = bs / 8;
                for(unsigned int i=0; i < numcontr; i++){
                    op_ptr rs, encr, h2_;
                    switch(ft){
                        case FT_TANH: 
                            rs   = row_select(x0_nl,x1_nl,m_enc);
                            encr = result(rs,2);
                            h2_  = 1.f-square(encr);  // tanh
                            break;
                        case FT_LOGISTIC:
                            rs   = row_select(x0_nl,x1_nl,m_enc); 
                            encr = result(rs,2);
                            h2_  = encr*(1.f-encr); // logistic
                            break;
                        case FT_NORM:
                            rs   = row_select(x0_nl,x1_nl,m_enc_lin); 
                            encr = result(rs,2);
                            h2_  = (1.f/norm) * pow(encr+0.001f, 1.f/norm - 1.f); // 1/2  1/sqrt(x), where m_enc=sqrt(x)
                            break;
                        case FT_ABS:
                            rs   = row_select(x0_nl,x1_nl, m_enc_lin, m_enc); 
                            h2_  = result(rs, 2) * pow(result(rs, 3), -1.f); // x * 1/sqrt(x^2), where m_enc=sqrt(x^2)
                            break;
                        case FT_1MEXPMX:
                            rs   = row_select(x0_nl,x1_nl,m_enc_lin); 
                            encr = result(rs,1);
                            h2_  = exp(-1.f,encr); // d/dx [1-exp(-x)]  == exp(-x)
                            break;
                    }
                    // derivative of first "activation function"
                    op_ptr h1_0, h1_1;
                    op_ptr x0r   = result(rs,0);
                    op_ptr x1r   = result(rs,1);
                    //       h1  = pow(x, norm)
                    //  -->  h1_ = norm * pow(x, norm-1)
                    if(norm == 2.f){
                        h1_0  = norm * x0r; 
                        h1_1  = norm * x1r;
                    }
                    else if(norm_is_even){
                        h1_0  = norm * pow(x0r, norm-1.f); 
                        h1_1  = norm * pow(x1r, norm-1.f);
                    }else if(norm == 1.f){
                        h1_0  = x0r * pow(abs(x0r), -1.f);
                        h1_1  = x1r * pow(abs(x1r), -1.f);
                    }else{
                        // note: do not cancel the 2.f, it does not work when x is negative.
                        h1_0  = norm * pow(pow(x0r,2.f), norm/2.f); 
                        h1_1  = norm * pow(pow(x1r,2.f), norm/2.f);
                    }

                    // derivative of initial non-linearity on linear feature-responses
                    //h1_0 = h1_0 * (1.f - pow(x0r,2.f));
                    //h1_1 = h1_1 * (1.f - pow(x1r,2.f));

                    //op_ptr tmp = sum( sum(pow(prod(mat_times_vec(m_weights_x,h1_,1), m_weights_h),2.f),0)*pow(h2_,2.f));
                    op_ptr tmp = 
                        sum( 
                            sum_to_vec(
                                pow(
                                    prod(mat_times_vec(m_weights_x0,h1_0,1), m_weights_x1),2.f)
                                ,1)
                            *pow(h1_1,2.f));
                    if(!m_reg_loss)
                        m_reg_loss = tmp;
                    else
                        m_reg_loss = m_reg_loss + tmp;
                }
                m_loss     = axpby(m_rec_loss, lambda/numcontr, m_reg_loss);
                m_reg_sink = sink("contractive loss", m_reg_loss);
            } 
            if(0){
                if (ft == FT_LOGISTIC || ft == FT_1MEXPMX){
                    // this assumes that m_enc is a probability, so it must be between 0 and 1!
                    float gamma = 1.0f;
                    m_reg_loss = mean(make_shared<BernoulliKullbackLeibler>(
                                0.10f,
                                (sum_to_vec(m_enc,1)/(float)bs)->result())); // soft L1-norm on hidden units
                    m_loss        = axpby(m_loss, gamma, m_reg_loss);
                }else{
                    // use mean squared loss, since m_enc is not a probability!
                    float gamma = 1.00f;
                    m_reg_loss = mean(m_enc); // soft L1-norm on hidden units
                    //op_ptr m_sparse_loss = mean(
                            //(sum(m_enc,0)/(float)bs)); // soft L1-norm on hidden units
                    m_loss        = axpby(m_loss, gamma, m_reg_loss);
                }
                m_reg_sink = sink("sparse loss", m_reg_loss);
            }
            if(0){
                // L2-weight decay
                m_loss = axpby(m_loss, 0.1f,sum(pow(m_weights_x,2.f)));
                // L1-weight decay
                //m_loss = axpby(m_loss, 0.00001f,sum(pow(pow(m_weights_x,2.f)+0.0001f,0.5f)));
            }
            m_loss_sink       = sink("total loss", m_loss);
            reset_weights();
        }
};

void load_batch(
        auto_encoder* ae,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    ae->input() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
}

//void visualize_filters(auto_encoder_rel* ae, zero_mean_unit_variance<>* normalizer, int fa,int fb, int image_size, int channels, unsigned int epoch){
void visualize_filters(auto_encoder_rel* ae, preprocessor<>* normalizer, int fa,int fb, int image_size, int channels, unsigned int epoch){
    if(epoch%10 != 0)
        return;
    //if(epoch%50 == 0)
        //cuv::libs::cimg::show(cuv::tensor<float,cuv::host_memory_space>(ae->m_weights_h->data()),"Wh");
    {
        std::string base = (boost::format("weights-%06d-")%epoch).str();
        // show the resulting filters
        //unsigned int n_rec = (bs>0) ? sqrt(bs) : 6;
        //cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', n_rec,n_rec, image_size,channels), "input");
        cuv::tensor<float,cuv::host_memory_space>  w = trans(ae->m_weights_x1->data().copy());
        std::cout << "Weight dims: "<<w.shape(0)<<", "<<w.shape(1)<<std::endl;
        auto wvis = arrange_filters(w, 'n', fa/2, fb, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', fa/2, fb, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
        if(normalizer){
            //ae->angle_stats(w);
            normalizer->reverse_transform(w); // no mean added
            wvis      = arrange_filters(w, 'n', fa/2, fb, image_size,channels,false);
            cuv::libs::cimg::save(wvis, base+"nr.png");
            wvis      = arrange_filters(w, 'n', fa/2, fb, image_size,channels,true);
            cuv::libs::cimg::save(wvis, base+"sr.png");
        }
    }
    {
        std::string base = (boost::format("veights-%06d-")%epoch).str();
        // show the resulting filters
        //unsigned int n_rec = (bs>0) ? sqrt(bs) : 6;
        //cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', n_rec,n_rec, image_size,channels), "input");
        matrix  w0 = ae->m_weights_x0->data().copy();
        matrix  w1 = ae->m_weights_x1->data().copy();
        cuvAssert(cuv::equal_shape(w0,w1));
        unsigned int hl = w0.shape(1);
        cuvnet::matrix w_(cuv::extents[w0.shape(0)][hl*2]);
        using namespace cuv;
        w_[indices[index_range()][index_range(0,hl)   ]] = w0;
        w_[indices[index_range()][index_range(hl,2*hl)]] = w1;
        cuv::tensor<float, cuv::host_memory_space> w = trans(w_);
        std::cout << "Weight dims: "<<w.shape(0)<<", "<<w.shape(1)<<std::endl;
        auto wvis = arrange_filters(w, 'n', hl/2, fa*fb/hl*2, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', hl/2, fa*fb/hl*2, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
        if(normalizer){
            //ae->angle_stats(w);
            normalizer->reverse_transform(w); // no mean added
            wvis      = arrange_filters(w, 'n', hl/2, fa*fb/hl*2, image_size,channels,false);
            cuv::libs::cimg::save(wvis, base+"nr.png");
            wvis      = arrange_filters(w, 'n', hl/2, fa*fb/hl*2, image_size,channels,true);
            cuv::libs::cimg::save(wvis, base+"sr.png");
        }
    }

    {
        std::string base = (boost::format("recons-%06d-")%epoch).str();
        cuv::tensor<float,cuv::host_memory_space> w = ae->m_decode_sink->cdata().copy();
        fa = sqrt(w.shape(0));
        fb = sqrt(w.shape(0));
        auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
        if(normalizer){
            normalizer->reverse_transform(w); 
            wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
            cuv::libs::cimg::save(wvis, base+"nr.png");
            wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
            cuv::libs::cimg::save(wvis, base+"sr.png");
        }
    }

    {
        std::string base = (boost::format("input-%06d-")%epoch).str();
        cuv::tensor<float,cuv::host_memory_space> w = ae->input().copy();
        fa = sqrt(w.shape(0));
        fb = sqrt(w.shape(0));
        auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
        if(normalizer){
            normalizer->reverse_transform(w); 
            wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
            cuv::libs::cimg::save(wvis, base+"nr.png");
            wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
            cuv::libs::cimg::save(wvis, base+"sr.png");
        }
    }
}

void dump_features(auto_encoder_rel* ae, 
        gradient_descent* gd,
        cuv::tensor<float,cuv::dev_memory_space>* alldata,
        unsigned int bs,
        unsigned int epoch)
{
    if(epoch % 100 != 0)
        return;
    std::ofstream os("features.dump");
    dumper dmp(ae->encoded());
    dmp.before_batch.connect(boost::bind(load_batch,ae,alldata,bs,_2));
    dmp.current_batch_num.connect(alldata->shape(0)/ll::constant(bs));
    dmp.dump(os);

    // after dumping data (with swiper obj of dumper), repair swiper obj of gradient descent!
    gd->repair_swiper();
}

void dump_weights(auto_encoder_rel* ae, unsigned int epoch)
{
    if(epoch % 100 != 0)
        return;
    cuvnet::tofile<float>("weights_x.npy", ae->m_weights_x->data());
    cuvnet::tofile<float>("weights_x0.npy", ae->m_weights_x0->data());
    cuvnet::tofile<float>("weights_x1.npy", ae->m_weights_x1->data());
}

int main(int argc, char **argv)
{
    cuv::initCUDA(0);
    cuv::initialize_mersenne_twister_seeds();
    cuvnet::Logger log;

    //------- dataset selection ---------
    natural_dataset ds_all("/home/local/datasets/natural_images");
    //cifar_dataset ds_all;
    //mnist_dataset ds_all("/home/local/datasets/MNIST");
    
    //------- main normalizer selection ---------
    pca_whitening normalizer(100,true,false, 0.01);
    //global_min_max_normalize<> normalizer;
    //identity_preprocessor<> normalizer;

    randomizer().transform(ds_all.train_data, ds_all.train_labels);
    splitter ds_split(ds_all,2);
    dataset ds  = ds_split[0];
    //ds.binary   = false;

    unsigned int fa=10,fb=10,bs=64;
    
    {   //-------------------------------------------------------------
        // pre-processing                                              +
        
        // subtract mean of each patch
        // http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing

        // Osindero & Hinton: First log, then ZMUV, then ZCA.
        // remark: weird: higher order whitening paper (CVPR'05) also whitens
        // in log-space /after/ regular whitening, retaining the sign
        zero_sample_mean<> n;
        log_transformer<> n2;
        zero_mean_unit_variance<> n3; 
        

        n2.fit_transform(ds.train_data); // results pretty much in gaussian
        n2.transform(ds.val_data);    // do the same to the validation set

        n.fit_transform(ds.train_data); // subtract sample mean
        n.transform(ds.val_data);

        n3.fit_transform(ds.train_data); // normalize each feature to get to defined range
        n3.transform(ds.val_data);    // do the same to the validation set
        
        normalizer.fit_transform(ds.train_data);
        normalizer.transform(ds.val_data);
        // end preprocessing                                           /
        //-------------------------------------------------------------
    }
    normalizer.write_params("pca_whitening");
        //auto_encoder_rel(unsigned int bs  , unsigned int inp1, unsigned int hl, unsigned int factorsize, bool binary, float noise=0.0f, float lambda=0.0f)
    auto_encoder_rel ae(bs, ds.train_data.shape(1), 50, fa*fb, ds.binary, 0.0f, 0.100f);
    dump_weights(&ae,0);

    std::vector<Op*> params = ae.unsupervised_params();

    Op::value_type alldata = bs==0 ? ds.val_data : ds.train_data;
    gradient_descent gd(ae.m_loss,0,params,0.05f,-0.00000f);
    gd.after_epoch.connect(0,boost::bind(&auto_encoder::print_loss, &ae, _1));
    gd.after_epoch.connect(0,boost::bind(&auto_encoder::reset_loss, &ae));
    gd.after_epoch.connect(0,boost::bind(visualize_filters,&ae,&normalizer,fa,fb,ds.image_size,ds.channels,_1));

    // TODO: dump_features kills gradient when epoch %100 ==0
    //gd.after_epoch.connect(1,boost::bind(dump_features,&ae,&gd,&alldata,bs,_1));
    gd.after_epoch.connect(1,boost::bind(dump_weights, &ae,_1));

    gd.before_batch.connect(boost::bind(load_batch,&ae,&alldata,bs,_2));
    gd.after_batch.connect(boost::bind(&auto_encoder::acc_loss, &ae));
    gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));

    gd.decay_learnrate(0.9995);
    gd.minibatch_learning(6000);

    
    return 0;
}
