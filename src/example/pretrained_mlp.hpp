#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

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
        acc_t    s_epochs;

        unsigned int    avg_epochs()  { 
            if(acc::count(s_epochs)==0)
                return 0;
            return acc::mean(s_epochs); 
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
