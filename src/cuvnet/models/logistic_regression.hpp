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
    private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                ar & boost::serialization::base_object<generic_regression>(*this);
            }
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

/**
 * logistic regression with L2 penalty.
 */
struct logistic_ridge_regression
: public logistic_regression
, public simple_weight_decay
{
    private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                ar & boost::serialization::base_object<generic_regression>(*this);
                ar & boost::serialization::base_object<simple_weight_decay>(*this);
            }
    public:
    typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * constructor forwards all except 1st arguments to logistic_regression.
     *
     * @param l2 regularization strength
     */
    template<typename... Params>
        logistic_ridge_regression(float l2, Params... args)
        : logistic_regression(args...)
        , simple_weight_decay(l2)
        {
        }
    /** default ctor for serialization */
    logistic_ridge_regression(){}
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
 * logistic regression with L1 penalty.
 */
struct logistic_lasso_regression
: public logistic_regression
, public lasso_regularizer
{
    private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                ar & boost::serialization::base_object<generic_regression>(*this);
                ar & boost::serialization::base_object<lasso_regularizer>(*this);
            }
    public:
    typedef boost::shared_ptr<Op>     op_ptr;
    /**
     * constructor forwards all except 1st arguments to logistic_regression.
     *
     * @param l1 regularization strength
     */
    template<typename... Params>
        logistic_lasso_regression(float l1, Params... args)
        : logistic_regression(args...)
        , lasso_regularizer(l1)
        {
        }
    /** default ctor for serialization */
    logistic_lasso_regression(){}
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


#endif /* __LOGISTIC_REGRESSION_HPP__ */
