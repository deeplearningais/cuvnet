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
#include <datasets/splitter.hpp>

#include <tools/crossvalid.hpp>
#include <tools/preprocess.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/visualization.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>


using namespace cuvnet;
namespace acc=boost::accumulators;
namespace ll = boost::lambda;

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
        void print_loss(unsigned int epoch) {
            std::cout << "epoch " << epoch<< "perf"<<acc::mean(s_total_loss)<<"reg"<<acc::mean(s_reg_loss)<<"rec"<<acc::mean(s_rec_loss) << std::endl;
            //g_worker->log(BSON("who"<<"AE"<<"epoch"<<epoch<<"perf"<<acc::mean(s_total_loss)<<"reg"<<acc::mean(s_reg_loss)<<"rec"<<acc::mean(s_rec_loss)));
            //g_worker->checkpoint();
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
        input_ptr    m_weights_x, m_weights_h;
        input_ptr    m_bias_xvis; /// bias of the reconstruction
        input_ptr    m_bias_h;    /// bias of the hidden layer
        sink_ptr     m_loss_sink, m_reg_sink, m_rec_sink, m_decode_sink;
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
            tmp += m_weights_x.get(), m_weights_h.get();
            tmp += m_bias_h.get();
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
            if(1)          s_total_loss((float)m_loss_sink->cdata()[0]);
            if(m_rec_sink) s_rec_loss  ((float)m_rec_sink->cdata()[0]);
            if(m_reg_sink) s_reg_loss  ((float)m_reg_sink->cdata()[0]);

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
        void print_loss(unsigned int epoch) {
            auto_encoder::print_loss(epoch);
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
                 m_weights_h.reset(new Input(cuv::extents[factorsize][hl],  "ae_wh" + boost::lexical_cast<std::string>(layer)));
                 m_bias_xvis.reset(new Input(cuv::extents[inp1],     "ae_bias_xvis" + boost::lexical_cast<std::string>(layer)));
                 m_bias_h   .reset(new Input(cuv::extents[hl],       "ae_bias_h"    + boost::lexical_cast<std::string>(layer)));
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
            ,m_weights_h(new Input(cuv::extents[factorsize][hl],  "ae_wh"))
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
                                    m(idx, i) = 1.f;
                            }
                        }
                    }else{
                        for(int i=0;i<fs;i++){
                            for(int j=0; j<step; j++){
                                int idx = i*step+j;
                                if(idx >= 0 && idx < hls)
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

            m_expected_size[0] = -1;
            m_expected_size[1] = -1;
            //normalize_columns(m_weights1->data(), m_expected_size[0]);
            //normalize_columns(m_weights2->data(), m_expected_size[1]);

            m_bias_xvis->data() = 0.f;
            m_bias_h->data()    = 0.f;
        }

    private:
        /**
         * initializes the functions in the AE  according to params given in the
         * constructor
         */
        void init(unsigned int bs  , unsigned int inp1, unsigned int hl, unsigned int factorsize, bool binary, float noise, float lambda) {
            Op::op_ptr corrupt               = m_input;
            if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
            if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);

            op_ptr x_ = prod(corrupt, m_weights_x);
            op_ptr xsq = pow(x_,2.f);
            m_enc     = logistic(mat_plus_vec( prod(xsq, m_weights_h), m_bias_h, 1));
            
            m_decode      = mat_plus_vec(prod(prod(m_enc, m_weights_h, 'n','t')*x_,m_weights_x,'n','t'),m_bias_xvis,1);
            m_decode_sink = sink("decoded", m_decode);
            m_rec_loss    = reconstruction_loss(m_input,m_decode);

            if(lambda>0.f) { // contractive AE
                op_ptr rs  = row_select(xsq,m_enc); // select same (random) row in h1 and h2
                op_ptr xsqr = result(rs,0);
                op_ptr encr = result(rs,1);

                op_ptr    h1_ = 2.f*xsq;
                op_ptr    h2_ = encr*(1.f-encr);

                m_reg_loss = sum( sum(pow(prod(mat_times_vec(m_weights_x,h1_,1), m_weights_h),2.f),0)*pow(h2_,2.f));
                m_loss        = axpby(m_rec_loss, lambda, m_reg_loss);
                m_rec_sink    = sink("reconstruction loss", m_rec_loss);
                m_reg_sink    = sink("contractive loss", m_reg_loss);
            } else{
                m_loss        = m_rec_loss; // no change
                m_rec_sink    = sink(m_rec_loss);
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

matrix trans(matrix& m){
    matrix mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}
//void visualize_filters(auto_encoder* ae, pca_whitening* normalizer, int fa,int fb, int image_size, int channels, unsigned int epoch){
void visualize_filters(auto_encoder_rel* ae, pca_whitening* normalizer, int fa,int fb, int image_size, int channels, unsigned int epoch){
    if(epoch%300 != 0)
        return;
    {
        std::string base = (boost::format("weights-%06d-")%epoch).str();
        // show the resulting filters
        //unsigned int n_rec = (bs>0) ? sqrt(bs) : 6;
        //cuv::libs::cimg::show(arrange_filters(ae.m_reconstruct->cdata(),'n', n_rec,n_rec, image_size,channels), "input");
        cuv::tensor<float,cuv::host_memory_space>  w = trans(ae->m_weights_x->data());
        std::cout << "Weight dims: "<<w.shape(0)<<", "<<w.shape(1)<<std::endl;
        auto wvis = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_filters(w, 'n', fa, fb, (int)sqrt(w.shape(1)),channels,true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
        normalizer->reverse_transform(w,true); // no mean added
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
        cuv::libs::cimg::save(wvis, base+"nr.png");
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
        cuv::libs::cimg::save(wvis, base+"sr.png");
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
        normalizer->reverse_transform(w); 
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
        cuv::libs::cimg::save(wvis, base+"nr.png");
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
        cuv::libs::cimg::save(wvis, base+"sr.png");
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
        normalizer->reverse_transform(w); 
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,false);
        cuv::libs::cimg::save(wvis, base+"nr.png");
        wvis      = arrange_filters(w, 'n', fa, fb, image_size,channels,true);
        cuv::libs::cimg::save(wvis, base+"sr.png");
    }
}

int main(int argc, char **argv)
{
    cuv::initCUDA(2);
    cuv::initialize_mersenne_twister_seeds();

    natural_dataset ds_all("/home/local/datasets/natural_images");
    pca_whitening normalizer(128,false,true, 0.01);
    splitter ds_split(ds_all,2);
    dataset ds  = ds_split[0];
    ds.binary   = false;

    unsigned int fa=16,fb=8,bs=512;
    
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
        //auto_encoder_rel(unsigned int bs  , unsigned int inp1, unsigned int hl, unsigned int factorsize, bool binary, float noise=0.0f, float lambda=0.0f)
    auto_encoder_rel ae(bs, ds.train_data.shape(1), 64, fa*fb, ds.binary, 0.0f, 0.0f);

    std::vector<Op*> params = ae.unsupervised_params();

    Op::value_type alldata = bs==0 ? ds.val_data : ds.train_data;
    gradient_descent gd(ae.m_loss,0,params,0.1f,-0.00000f);
    gd.after_epoch.connect(boost::bind(&auto_encoder::print_loss, &ae, _1));
    gd.after_epoch.connect(boost::bind(&auto_encoder::reset_loss, &ae));
    gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&normalizer,fa,fb,ds.image_size,ds.channels,_1));
    gd.before_batch.connect(boost::bind(load_batch,&ae,&alldata,bs,_2));
    gd.after_batch.connect(boost::bind(&auto_encoder::acc_loss, &ae));
    gd.current_batch_num.connect(ds.train_data.shape(0)/ll::constant(bs));

    if(bs==0){
        ae.input()= alldata;
        alldata.dealloc();
        gd.batch_learning(3200);
    }
    else      gd.minibatch_learning(6000);
    
    return 0;
}
