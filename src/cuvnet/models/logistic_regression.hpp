#ifndef __CUVNET_LOGREG_HPP__
#     define __CUVNET_LOGREG_HPP__

#include <boost/serialization/version.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/noiser.hpp>
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
            typedef boost::shared_ptr<Sink> sink_ptr;

            op_ptr m_loss, m_classloss, m_estimator;
            input_ptr m_W;
            input_ptr m_bias;
            op_ptr m_X;
            input_ptr m_Y;

            sink_ptr m_estimator_sink;

            boost::shared_ptr<Noiser> m_noiser;

            /// default ctor for serialization.
            logistic_regression(){}

            /**
             * logistic regression ctor using one-out-of-n encoding in Y.
             * @param X the estimator input
             * @param Y the target value
             * @param degenerate if true, assume is already estimator
             * @param dropout if true, apply dropout to inputs
             */
            logistic_regression(op_ptr X, op_ptr Y, bool degenerate=false, bool dropout=false);

            /**
             * logistic regression ctor using class id coding in Y.
             * @param X the estimator input
             * @param Y the target value
             * @param n_classes the number of classes (ignored if degenerate is true)
             * @param dropout if true, apply dropout to inputs
             */
            logistic_regression(op_ptr X, op_ptr Y, int n_classes, bool dropout=false);

            virtual std::vector<Op*> get_params();
            virtual void reset_params();
            virtual op_ptr loss()const;
            virtual op_ptr error()const;

            virtual void set_predict_mode(bool b=true);

            virtual ~logistic_regression(){}
            private:
            friend class boost::serialization::access;                                                                 
            template<class Archive>                                                                                    
                void serialize(Archive& ar, const unsigned int version) { 
                    ar & boost::serialization::base_object<model>(*this);;
                    ar & m_loss & m_classloss & m_W & m_bias & m_estimator;
                    ar & m_X & m_Y;
                    if(version > 0)
                        ar & m_estimator_sink;
                    if(version > 1)
                        ar & m_noiser;
                };
        };

    }
}
BOOST_CLASS_EXPORT_KEY(cuvnet::models::logistic_regression) 
BOOST_CLASS_VERSION(cuvnet::models::logistic_regression, 2);
#endif /* __CUVNET_LOGREG_HPP__ */
