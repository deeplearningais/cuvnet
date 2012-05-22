#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <cuv.hpp>

#include <cuvnet/ops.hpp>

using namespace cuvnet;
namespace acc=boost::accumulators;

typedef boost::shared_ptr<Input>  input_ptr;
typedef boost::shared_ptr<Sink> sink_ptr;
typedef boost::shared_ptr<Op>     op_ptr;

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
        acc_t    s_iters; // iterations this was trained for

        unsigned int    avg_iters()  { 
            if(acc::count(s_iters)==0)
                return 0;
            return acc::mean(s_iters); 
        }
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
        float perf_loss(){
            return m_loss_sum / m_loss_sum_cnt;
        }
        float perf() {
            return m_class_err/m_class_err_cnt;
        }
        void reset_loss() {
            m_loss_sum = m_class_err = m_class_err_cnt = m_loss_sum_cnt = 0;
        }
        void log_loss(const char* what, unsigned int epoch) {
            mongo::BSONObjBuilder bob;
            bob<<"who"<<"mlp"<<"epoch"<<epoch<<"type"<<what;
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
         * @param wd weight decay of output layer
         */
        pretrained_mlp(op_ptr inputs, unsigned int outputs, bool softmax, float wd)
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
                    m_loss = mean(sum_to_vec(pow(axpby(-1.f,m_targets,m_output),2.f),0));
                if(wd>0.f)
                    m_loss = axpby(m_loss,wd,mean(pow(m_weights,2.f)));
                m_loss_sink = sink("MLP loss",m_loss);

                reset_weights();
            }
        /// initialize weights and biases
        void reset_weights() {
            // this is logistic regression, no symmetrie breaking required!
            m_weights->data() =   0.f;
            m_bias->data()   = 0.f;
            m_bias->data()   = 0.f;
        }
};
