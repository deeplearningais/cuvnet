#include "initialization.hpp"
#include "linear_regression.hpp"
#include <cuvnet/ops.hpp>
#include <cuvnet/op_utils.hpp>

namespace cuvnet
{
    namespace models
    {
        void linear_regression::reset_params(){
            if(m_W)
                initialize_dense_glorot_bengio(m_W, false);
            if(m_bias) {
                m_bias->data() = 0.f;
                m_bias->set_weight_decay_factor(0.f);
            }
        }
        linear_regression::op_ptr linear_regression::loss()const{ return m_loss; }

        linear_regression::linear_regression(op_ptr X, op_ptr Y, bool degenerate, bool dropout){
            if(dropout)
                X = m_noiser = zero_out(X, 0.5f);

            determine_shapes(*Y);
            determine_shapes(*X);

            op_ptr estimator;
            if(!degenerate) {
                m_W = input(cuv::extents[X->result()->shape[1]][Y->result()->shape[1]]);
                m_bias = input(cuv::extents[Y->result()->shape[1]]);
                estimator = mat_plus_vec(prod(X, m_W), m_bias, 1);
            }
            else
                estimator = X;

            m_loss = mean(sum_to_vec(square(estimator-Y), 0));
        }
        void linear_regression::set_predict_mode(bool b){
            if(m_noiser)
                m_noiser->set_active(!b);
        }
        std::vector<Op*> linear_regression::get_params(){
            std::vector<Op*> params;
            if(m_W)    params.push_back(m_W.get());
            if(m_bias) params.push_back(m_bias.get());
            return params;
        }
    }
}
