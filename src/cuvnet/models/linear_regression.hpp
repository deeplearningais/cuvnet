#ifndef __LINEAR_REGRESSION_HPP__
#     define __LINEAR_REGRESSION_HPP__

#include <cuvnet/models/generic_regression.hpp>
#include <cuvnet/ops.hpp>

namespace cuvnet
{

/**
 * implements mean square error loss.
 *
 * \f$ L( \hat{y}, y) = \sum{i=1}{N} (\hat{y} - y)^2 \f$
 *
 * @ingroup models
 */
class linear_regression:  public generic_regression{
    public:

     /**
      * Constructor
      *
      * @param input a function that generates the input 
      * @param target a function that generates the target
      */
    linear_regression(op_ptr input, op_ptr target): generic_regression(input, target){}
   

    protected:
    
    /**
     * Loss function
     *  @return a function that calculates the logistic loss 
     */
    op_ptr loss(){
          return mean( sum_to_vec(pow(axpby(get_target(), -1.f, get_estimator()), 2.f), 0) );
    }

};


}


#endif /* __LINEAR_REGRESSION_HPP__ */
