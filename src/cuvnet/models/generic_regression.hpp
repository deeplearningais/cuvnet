#ifndef __GENERIC_REGRESSION_HPP__
#     define __GENERIC_REGRESSION_HPP__

#include <boost/tuple/tuple.hpp>
#include <boost/assign.hpp>
#include <cuv.hpp>
#include <cuvnet/ops.hpp>

namespace cuvnet
{
/**
 * Abstract base class for regression. 
 *
 * The loss function \f$ L( \hat{y}, y) \f$ must be implemented by derived
 * models and should be minimal when \f$ \hat{y} = y\f$. 
 *
 * @ingroup models
 */
class generic_regression 
{
    public:
        /** 
         * this is the type of a `function', eg the output or the loss.
         */
        typedef boost::shared_ptr<Op> op_ptr;
        typedef boost::shared_ptr<ParameterInput> input_ptr; 
    protected:
       op_ptr m_input;    ///< x 
       op_ptr m_target;   ///< y
       op_ptr m_user_loss;       ///<  user-supplied loss function
       op_ptr m_regularization_loss;       ///< regularization loss

    private:
       input_ptr m_weights;  ///< W 
       input_ptr m_bias;     ///< b_y
       op_ptr    m_loss;     ///< accumulated loss (user-supplied plus regularization)
       op_ptr    m_est;      ///< \f$ \hat{y} = x W + b_y \f$

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
            m_weights.reset(new ParameterInput(cuv::extents[input_dim][m_target_dim],"regression_weights"));
            m_bias.reset(new ParameterInput(cuv::extents[m_target_dim],             "regression_bias"));

            // inits weights with random numbers,sets bias to zero
            reset_weights();
            
        }

        /**
         * Returns the (additive) regularizer for the auto-encoder.
         * Defaults to no regularization.
         */
        virtual boost::tuple<float,op_ptr> regularize(){
            return boost::make_tuple(0.f,op_ptr());
        }

        /**
         * gets the loss function, it initialize it if it is not already initialized 
         * @return loss function
         */
        op_ptr get_loss(){
            if(!m_loss){
                m_user_loss = loss();
                float lambda;
                boost::tie(lambda, m_regularization_loss) = regularize();
                if(lambda && m_regularization_loss) 
                    m_loss = axpby(m_user_loss, lambda, m_regularization_loss);
                else
                    m_loss = m_user_loss;
            }
            return m_loss;
        }

        /**
         * initialize the weights with random numbers
         * @overload
         */
        virtual void reset_weights(){
            m_weights->data() = 0.f;
            m_bias->data()   = 0.f;
            m_bias->m_weight_decay_factor = 0.f; // do not apply wd to bias
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
         * Returns the linear part of the estimator.
         * \f$  x W + b_y \f$
         */
        op_ptr  get_estimator(){ 
            if(!m_est)
                m_est = mat_plus_vec(
                        prod( m_input, m_weights)
                        ,m_bias,1);
            return m_est;
        }
        

        /**
         * @return the input
         */
        input_ptr get_weights(){
            return m_weights;
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
            return classification_loss(sink(m_est), m_target);
        }

       protected:

       /**
        * Loss function
        *  @return a function that calculates the loss 
        */
       virtual op_ptr loss() = 0;
};




}


#endif /* __GENERIC_REGRESSION_HPP__ */
