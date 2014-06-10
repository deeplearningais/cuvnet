#include "initialization.hpp"
#include "logistic_regression.hpp"
#include <cuvnet/ops.hpp>
#include <cuvnet/op_utils.hpp>

namespace cuvnet
{
    namespace models
    {
        void logistic_regression::reset_params(){
            //if(m_W)
                //initialize_dense_glorot_bengio(m_W, false);
            if(m_W)
                // might not exist if this is a 'degenerate' logreg
                m_W->data() = 0.f;
            if(m_bias) {
                m_bias->data() = 0.f;
                m_bias->set_weight_decay_factor(0.f);
            }
        }
        logistic_regression::op_ptr logistic_regression::loss()const{ return m_loss; }
        logistic_regression::op_ptr logistic_regression::error()const{ return m_classloss; }

        logistic_regression::logistic_regression(op_ptr X, op_ptr Y, bool degenerate){
            determine_shapes(*X);
            determine_shapes(*Y);
            m_X = X;
            m_Y = boost::dynamic_pointer_cast<ParameterInput>(Y);

            if(!degenerate) {
                m_W = input(cuv::extents[X->result()->shape[1]][Y->result()->shape[1]], "logreg_W");
                m_bias = input(cuv::extents[Y->result()->shape[1]], "logreg_b");
                m_estimator = mat_plus_vec(prod(X, m_W), m_bias, 1);
                m_estimator_sink = sink("logreg_est", m_estimator);
            }
            else
                m_estimator = X;

            // TODO: if Y is vector, switch to non-multinomial logreg!
            m_loss = mean(
                    multinomial_logistic_loss(
                        m_estimator, Y, 1));
            m_classloss = classification_loss(m_estimator, Y);
        }
        logistic_regression::logistic_regression(op_ptr X, op_ptr Y, int n_classes){
            determine_shapes(*X);
            determine_shapes(*Y);
            m_X = X;
            m_Y = boost::dynamic_pointer_cast<ParameterInput>(Y);

            m_W = input(cuv::extents[X->result()->shape[1]][n_classes], "logreg_W");
            m_bias = input(cuv::extents[n_classes], "logreg_b");
            m_estimator = mat_plus_vec(prod(X, m_W), m_bias, 1);
            m_estimator_sink = sink("logreg_est", m_estimator);

            // TODO: if Y is vector, switch to non-multinomial logreg!
            m_loss = 
                    multinomial_logistic_loss2(m_estimator, Y);
            m_classloss = result(m_loss, 2);
        }
        std::vector<Op*> logistic_regression::get_params(){
            std::vector<Op*> params;
            if(m_W)    params.push_back(m_W.get());
            if(m_bias) params.push_back(m_bias.get());
            return params;
        }
    }
}
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::logistic_regression);
