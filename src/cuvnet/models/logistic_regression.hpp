#ifndef __CUVNET_LOGREG_HPP__
#     define __CUVNET_LOGREG_HPP__

#include <cuvnet/ops/input.hpp>
#include "models.hpp"

namespace cuvnet
{
    namespace models
    {
        /**
         * implements (multinomial) logistic regression.
         */
        struct logistic_regression : public model{

            typedef boost::shared_ptr<Op> op_ptr;
            typedef boost::shared_ptr<ParameterInput> input_ptr;

            op_ptr m_loss, m_classloss, m_estimator;
            input_ptr m_W;
            input_ptr m_bias;


            /// default ctor for serialization.
            logistic_regression(){}

            /**
             * logistic regression ctor.
             * @param X the estimator input
             * @param Y the target value
             * @param degenerate if true, assume is already estimator
             */
            logistic_regression(op_ptr X, op_ptr Y, bool degenerate=false);

            virtual std::vector<Op*> get_params();
            virtual void reset_params();
            virtual op_ptr loss()const;
            virtual op_ptr error()const;

            virtual ~logistic_regression(){}
            private:
            friend class boost::serialization::access;                                                                 
            template<class Archive>                                                                                    
                void serialize(Archive& ar, const unsigned int version) { 
                    ar & boost::serialization::base_object<boost::enable_shared_from_this<model> >(*this);;
                    ar & m_loss & m_classloss & m_W & m_bias & m_estimator;
                };
        };
    }
}
#endif /* __CUVNET_LOGREG_HPP__ */
