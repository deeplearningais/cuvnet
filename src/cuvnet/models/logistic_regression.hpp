#ifndef __LOGISTIC_REGRESSION_HPP__
#     define __LOGISTIC_REGRESSION_HPP__

#include <cuvnet/models/simple_auto_encoder.hpp> /* for no_regularization */
#include <cuvnet/models/generic_regression.hpp>
#include <cuvnet/ops.hpp>

namespace cuvnet
{

/**
 * implements multinomial logistic regression.
 *
 * \f[ L( \hat{y}, y) = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (\hat{y_{n,k}} - ln(\sum_{j}e^{y_{n,j}})) \f] 
 *
 * @ingroup models
 */
class logistic_regression:  public generic_regression{
    public:
        typedef boost::shared_ptr<Op> op_ptr;
    
     /**
      * Constructor
      *
      * @param input a function that generates the input 
      * @param target a function that generates the target
      */   
    logistic_regression(op_ptr input, op_ptr target): generic_regression(input, target){
    }

     /**
      * Default Constructor: You need to call init() to finish initialization.
      */   
    logistic_regression(){ }

    protected:

    /**
     * Loss function
     *  @return a function that calculates the logistic loss 
     */
    op_ptr loss(){
         return mean(multinomial_logistic_loss(get_estimator(), get_target(),1));
    }

};


}


#endif /* __LOGISTIC_REGRESSION_HPP__ */
