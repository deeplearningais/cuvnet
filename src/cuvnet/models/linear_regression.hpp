#ifndef __CUVNET_LINREG_HPP__
#     define __CUVNET_LINREG_HPP__

#include <cuvnet/ops/input.hpp>
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


            /// default ctor for serialization.
            linear_regression(){}

            /**
             * linear regression ctor.
             * @param X the estimator input
             * @param Y the target value
             * @param degenerate if true, assume is already estimator
             */
            linear_regression(op_ptr X, op_ptr Y, bool degenerate=false);

            virtual std::vector<Op*> get_params();
            virtual void reset_params();
            virtual op_ptr loss()const;

            virtual ~linear_regression(){}
            private:
            friend class boost::serialization::access;                                                                 
            template<class Archive>                                                                                    
                void serialize(Archive& ar, const unsigned int version) { 
                    ar & boost::serialization::base_object<boost::enable_shared_from_this<model> >(*this);;
                    ar & m_loss & m_W & m_bias;
                };
        };
    }
}
#endif /* __CUVNET_LINREG_HPP__ */
