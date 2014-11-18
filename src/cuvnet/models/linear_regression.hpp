#ifndef __CUVNET_LINREG_HPP__
#     define __CUVNET_LINREG_HPP__

#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/noiser.hpp>
#include "models.hpp"

namespace cuvnet
{
    namespace models
    {
        /**
         * implements (multi-target) linear regression.
         */
        struct linear_regression : public model{

            typedef boost::shared_ptr<Op> op_ptr;
            typedef boost::shared_ptr<ParameterInput> input_ptr;

            op_ptr m_loss;
            input_ptr m_W;
            input_ptr m_bias;
            boost::shared_ptr<Noiser> m_noiser;


            /// default ctor for serialization.
            linear_regression(){}

            /**
             * linear regression ctor.
             * @param X the estimator input
             * @param Y the target value
             * @param degenerate if true, assume is already estimator
             * @param dropout if true, do dropout in inputs before multiplication
             */
            linear_regression(op_ptr X, op_ptr Y, bool degenerate=false, bool dropout=false);

            virtual std::vector<Op*> get_params();
            virtual void reset_params();
            virtual op_ptr loss()const;

            virtual void set_predict_mode(bool b=true);

            virtual ~linear_regression(){}

            op_ptr m_estimator, m_Y, m_X;

            private:
            friend class boost::serialization::access;                                                                 
            template<class Archive>                                                                                    
                void serialize(Archive& ar, const unsigned int version) { 
                    ar & boost::serialization::base_object<model>(*this);;
                    ar & m_loss & m_W & m_bias;
                    if(version > 0)
                        ar & m_noiser;
                    if(version > 1)
                        ar & m_estimator & m_X & m_Y;
                };
        };
    }
}
BOOST_CLASS_VERSION(cuvnet::models::linear_regression, 2);
#endif /* __CUVNET_LINREG_HPP__ */
