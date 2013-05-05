#ifndef __CUVNET_LOGREG_HPP__
#     define __CUVNET_LOGREG_HPP__

#include <cuvnet/ops/input.hpp>
#include "models.hpp"

namespace cuvnet
{
    namespace models
    {
        struct logistic_regression : public model{

            typedef boost::shared_ptr<Op> op_ptr;
            typedef boost::shared_ptr<ParameterInput> input_ptr;

            op_ptr m_loss, m_classloss;
            input_ptr m_W;
            input_ptr m_bias;

            virtual void model::get_params(){
                std::vector<Op*> params(2);
                m_W.
            }

            /// default ctor for serialization.
            logistic_regression(){
            }
            logistic_regression(op_ptr X, op_ptr Y){
                determine_shapes(*X);
                determine_shapes(*Y);
                m_W = input(cuv::extents[X->result()->shape[1]][Y->result()->shape[1]]);
                m_bias = input(cuv::extents[Y->result()->shape[1]]);

                //std::cerr << "WARNING double softmax!" << std::endl;
                //op_ptr estimator = softmax(mat_plus_vec(prod(X,m_W), m_bias, 1), 1);
#if OUTPUT_CLASSIFIER_HAS_HIDDEN
                op_ptr estimator = mat_plus_vec(prod(X,m_W), m_bias, 1);
#else
                op_ptr estimator = X;
#endif
                m_loss = mean(
                        multinomial_logistic_loss(
                            estimator, Y, 1));
                m_classloss = classification_loss(estimator, Y);
            }

            void reset_weights(){
                {
                    float wnorm = m_W->data().shape(0)
                        +         m_W->data().shape(1);
                    float diff = std::sqrt(6.f/wnorm);
                    diff *= 4.f; // for logistic activation function "only"
                    cuv::fill_rnd_uniform(m_W->data());
                    m_W->data() *= diff*2.f;
                    m_W->data() -= diff;
                }

                m_bias->data() = 0.f;
                m_bias->set_weight_decay_factor(0.f);
            }

            op_ptr loss(){ return m_loss; }
            op_ptr classification_error(){return m_classloss;}

            private:
            friend class boost::serialization::access;                                                                 
            template<class Archive>                                                                                    
                void serialize(Archive& ar, const unsigned int version) { 
                    ar & m_loss & m_classloss & m_W & m_bias;
                };
        };
    }
}
#endif /* __CUVNET_LOGREG_HPP__ */
