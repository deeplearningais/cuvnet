#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuvnet/ops.hpp>

using namespace cuvnet;
using boost::make_shared;
namespace acc=boost::accumulators;

typedef
acc::accumulator_set<double,
    acc::stats<acc::tag::mean, acc::tag::variance(acc::lazy) > > acc_t;

struct auto_encoder{
    boost::shared_ptr<Input>  m_input, m_weights,m_bias_h,m_bias_y;
    boost::shared_ptr<Sink> m_out, m_reconstruct, m_corrupt;
    boost::shared_ptr<Op>     m_decode, m_enc;
    boost::shared_ptr<Op>     m_loss, m_rec_loss, m_contractive_loss;
    acc_t                     s_loss;
    matrix&       input() {return m_input->data();}
    const matrix& output(){return m_out->cdata();}

    void print_loss(unsigned int epoch){ std::cout << epoch<<": "<<acc::mean(s_loss)<<std::endl;}
    void acc_loss(){ s_loss((float) output()[0]);}
    void reset_loss(){ s_loss = acc_t();}

    auto_encoder(unsigned int inp0, unsigned int inp1, unsigned int hl, bool binary, float noise=0.0f, float lambda=0.0f)
    :m_input(new Input(cuv::extents[inp0][inp1],"input"))
    ,m_weights(new Input(cuv::extents[inp1][hl],"weights"))
    ,m_bias_h(new Input(cuv::extents[hl],       "bias_h"))
    ,m_bias_y(new Input(cuv::extents[inp1],     "bias_y"))
    {
        Op::op_ptr corrupt;
        if(0);
        else if( binary && noise>0.f) corrupt =       zero_out(m_input,noise);
        else if(!binary && noise>0.f) corrupt = add_rnd_normal(m_input,noise);
        else corrupt = m_input;
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
        m_out         = make_shared<Sink>(m_rec_loss->result()); // reconstruction error
        m_reconstruct = make_shared<Sink>(m_decode->result());   // for visualization of reconstructed images
        //m_corrupt = make_shared<Sink>(corrupt->result());      // for visualization of corrupted     images

        if(lambda>0.f){ // contractive AE
            m_contractive_loss = 
                sum(sum(pow(m_enc*(1.f-m_enc),2.f),0) 
                        * sum(pow(m_weights,2.f),0));
            m_loss        = axpby(m_rec_loss, lambda/(float)inp0, m_contractive_loss);
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

