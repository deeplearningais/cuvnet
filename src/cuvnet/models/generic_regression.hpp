#ifndef __REGRESSION_HPP__
#     define __REGRESSION_HPP__

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuv.hpp>

#include <cuvnet/ops.hpp>

namespace cuvnet
{
/*
 * This is the base class for regression. The loss function \f$ L( \hat{y}, y) \f$ is minimal when \f$ \hat{y} = y\f$. 
 */
class generic_regression 
{
    public:
        /** 
         * this is the type of a `function', e.g. the output or the loss 
         */
        typedef boost::shared_ptr<Op> op_ptr;
        typedef boost::shared_ptr<Input> input_ptr; 
    protected:
       op_ptr m_input;    ///< x 
       op_ptr m_target;   ///< y
       input_ptr m_weights;  ///< W 
       input_ptr m_bias;     ///< b_y
       op_ptr m_loss;       ///< loss
       op_ptr m_est;      ///< \f$ \hat{y} = x W + b_y \f$

    public: 
       
        /**
         * Constructor
         *
         * @param input a function that generates the input
         * @param target a function that generates the target
         */
        generic_regression(op_ptr input, op_ptr target)
            : m_input(input)
            , m_target(target)
        {
            // initialize the weights and bias 
            m_input->visit(determine_shapes_visitor()); 
            unsigned int input_dim = m_input->result()->shape[1]; 
            m_target->visit(determine_shapes_visitor());
            unsigned int m_target_dim = target->result()->shape[1];  
            m_weights.reset(new Input(cuv::extents[input_dim][m_target_dim],"weights"));
            m_bias.reset(new Input(cuv::extents[m_target_dim],             "bias"));

            m_est      = estimator(input);

            // inits weights with random numbers,sets bias to zero
            reset_weights();
            
        }

        /**
         * gets the loss function, it initialize it if it is not already initialized 
         * @return loss function
         */
        op_ptr get_loss(){
            if(!m_loss)
                m_loss = loss();
            return m_loss;
        }

        /**
         * initialize the weights with random numbers
         * @overload
         */
        virtual void reset_weights(){
            m_weights->data() = 0.f;
            m_bias->data()   = 0.f;
        }

        /**
         * Determine the parameters 
         */
        virtual std::vector<Op*> params(){
            return boost::assign::list_of(m_weights.get())(m_bias.get());
        };

        /**
         * Destructor
         */
        virtual ~generic_regression(){}

        /**
         * @return the input
         */
        op_ptr get_estimator(){return m_est;}

        /// \f$  x W + b_y \f$
        op_ptr  estimator(op_ptr& input){ 
            return mat_plus_vec(
                prod( input, m_weights)
                ,m_bias,1);
        }

        
        /**
         * @return the target
         */
        op_ptr get_target(){
            return m_target;
        }
        

        /**
         * @return the classification error 
         */
        op_ptr classification_error(){
            return classification_loss(m_est, m_target);
        }

       protected:

       /**
        * Loss function
        *  @return a function that calculates the loss 
        */
       virtual op_ptr loss() = 0;
};

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

/**
 * implements mean square error loss \f$ L( \hat{y}, y) = \sum{i=1}{N} (\hat{y} - y)^2 \f$
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


#endif /* __GENERIC_REGRESSION_HPP__ */
