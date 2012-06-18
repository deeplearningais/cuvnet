#ifndef __LOGISTIC_REGRESSION_HPP__
#     define __LOGISTIC_REGRESSION_HPP__

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <cuvnet/models/generic_regression.hpp>
#include <cuv.hpp>

#include <cuvnet/ops.hpp>

namespace cuvnet
{

/**
 * implements multinomial logistic regression \f[ L( \hat{y}, y) = - \frac{1}{N} \sum{i=1}{N} \sum{k=1}{K} (\hat{y_{n,k}} - ln(\sum_{j}e^{y_{n,j}})) \f] 
 */
class logistic_regression:  public generic_regression{
    public:
    
     /**
      * Constructor
      *
      * @param input a function that generates the input 
      * @param target a function that generates the target
      */   
    logistic_regression(op_ptr input, op_ptr target): generic_regression(input, target){
    }

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
