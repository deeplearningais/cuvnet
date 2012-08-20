#ifndef __GENERIC_REGRESSION_HPP__
#     define __GENERIC_REGRESSION_HPP__

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
template<class Base>
class generic_regression 
: public Base
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
       input_ptr m_weights;  ///< W 
       input_ptr m_bias;     ///< b_y
       op_ptr m_loss;       ///< accumulated loss (user-supplied plus regularization)
       op_ptr m_user_loss;       ///<  user-supplied loss function
       op_ptr m_regularization_loss;       ///< regularization loss
       op_ptr m_est;      ///< \f$ \hat{y} = x W + b_y \f$
       float m_regularization_strength; /// constant with which regularizer is multiplied

    public: 
       
        /**
         * Constructor
         *
         * @param input a function that generates the input
         * @param target a function that generates the target
         * @param regularization_strength value with which regularization loss is multiplied
         */
        generic_regression(op_ptr input, op_ptr target, float regularization_strength = 0.f)
            : m_input(input)
            , m_target(target)
            , m_regularization_strength(regularization_strength)
        {
            // initialize the weights and bias 
            m_input->visit(determine_shapes_visitor()); 
            unsigned int input_dim = m_input->result()->shape[1]; 
            m_target->visit(determine_shapes_visitor());
            unsigned int m_target_dim = target->result()->shape[1];  
            m_weights.reset(new ParameterInput(cuv::extents[input_dim][m_target_dim],"regression_weights"));
            m_bias.reset(new ParameterInput(cuv::extents[m_target_dim],             "regression_bias"));

            m_est      = estimator(input);

            // inits weights with random numbers,sets bias to zero
            reset_weights();
            
        }

        /**
         * gets the loss function, it initialize it if it is not already initialized 
         * @return loss function
         */
        op_ptr get_loss(){
            if(!m_loss){
                m_user_loss = loss();
                if(m_regularization_strength > 0.f)
                {
                    m_regularization_loss = Base::regularize(params());
                    if(m_regularization_loss) 
                        m_loss = m_user_loss + m_regularization_strength * m_regularization_loss;
                    else
                        m_loss = m_user_loss;
                }else
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
