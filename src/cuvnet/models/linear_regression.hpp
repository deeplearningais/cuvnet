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
        typedef boost::shared_ptr<Op>     op_ptr;

     /**
      * Constructor
      *
      * @param input a function that generates the input 
      * @param target a function that generates the target
      */
    linear_regression(op_ptr input, op_ptr target): generic_regression(input, target){}

     /**
      * Default Constructor: You need to call init() to finish initialization.
      */
    linear_regression(){}
   

    protected:
    
    /**
     * Loss function
     *  @return a function that calculates the logistic loss 
     */
    op_ptr loss(){
          return mean( sum_to_vec(pow(axpby(get_target(), -1.f, get_estimator()), 2.f), 0) );
    }

};

/**
 * linear regression with L2 penalty.
 */
struct linear_ridge_regression
: public linear_regression
, public simple_weight_decay
{
    typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * constructor forwards all except 1st arguments to linear_regression.
     *
     * @param l2 regularization strength
     */
    template<typename... Params>
        linear_ridge_regression(float l2, Params... args)
        : linear_regression(args...)
        , simple_weight_decay(l2)
        {
        }
    /**
     * Returns the L2 regularization loss.
     */
    virtual boost::tuple<float,op_ptr> regularize(){
        return boost::make_tuple(
                simple_weight_decay::strength(),
                simple_weight_decay::regularize(params()));
    }
};

/**
 * linear regression with L1 penalty.
 */
struct linear_lasso_regression
: public linear_regression
, public lasso_regularizer
{
    typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * constructor forwards all except 1st arguments to linear_regression.
     *
     * @param l1 regularization strength
     */
    template<typename... Params>
        linear_lasso_regression(float l1, Params... args)
        : linear_regression(args...)
        , lasso_regularizer(l1)
        {
        }
    /**
     * Returns the L1 regularization loss.
     */
    virtual boost::tuple<float,op_ptr> regularize(){
        return boost::make_tuple(
                lasso_regularizer::strength(),
                lasso_regularizer::regularize(params()));
    }
};

}


#endif /* __LINEAR_REGRESSION_HPP__ */
