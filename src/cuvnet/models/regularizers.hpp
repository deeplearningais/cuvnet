#ifndef __CUVNET_REGULARIZERS_HPP__
#     define __CUVNET_REGULARIZERS_HPP__
#include <cuvnet/ops.hpp>

namespace cuvnet
{

/**
 * Implements a weight decay to be mixed into a simple auto encoders or logistic regression.
 *
 * @ingroup models
 */
struct simple_weight_decay
{
    private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                ar & m_strength;
            }
    public:
    float m_strength;

    simple_weight_decay(float strength=0.f):m_strength(strength){}
    typedef boost::shared_ptr<Op>     op_ptr;

    /**
     * Returns the regularization strength.
     */
    inline float strength(){ return m_strength; }
    /**
     * L2 regularization of all two or more-dimensional ParameterInput objects
     */
    op_ptr regularize(const std::vector<Op*>& unsupervised_params){ 
        op_ptr regloss;
        int cnt = 0;
        BOOST_FOREACH(Op* op, unsupervised_params){
            if(dynamic_cast<ParameterInput*>(op)->data().ndim()<=1)
                continue;

            op_ptr tmp = mean(pow(op->shared_from_this(),2.f)); 
            if(cnt == 0)
                regloss = tmp;
            else if(cnt == 1)
                regloss = regloss + tmp;
            else
                regloss = add_to_param(regloss, tmp);
            cnt ++ ;
        }
        return regloss;
    }
};

/**
 * Implements a L1 weight decay to be mixed into a simple auto encoders or logistic regression.
 *
 * @ingroup models
 */
struct lasso_regularizer
{
    private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                ar & m_strength;
            }
    public:
    float m_strength;

    lasso_regularizer(float strength=0.f):m_strength(strength){}
    typedef boost::shared_ptr<Op>     op_ptr;

    /**
     * Returns the regularization strength.
     */
    inline float strength(){ return m_strength; }
    /**
     * L1 regularization of all two or more-dimensional ParameterInput objects.
     *
     * @note this does not produce exact zeros when used together with
     * gradient_descent. The region around 0 is approximated with a squared
     * function to ensure differentiability.
     */
    op_ptr regularize(const std::vector<Op*>& unsupervised_params){ 
        op_ptr regloss;
        int cnt = 0;
        BOOST_FOREACH(Op* op, unsupervised_params){
            if(dynamic_cast<ParameterInput*>(op)->data().ndim()<=1)
                continue;

            op_ptr tmp = mean(abs(op->shared_from_this())); 
            if(cnt == 0)
                regloss = tmp;
            else if(cnt == 1)
                regloss = regloss + tmp;
            else
                regloss = add_to_param(regloss, tmp);
            cnt ++ ;
        }
        return regloss;
    }
};

}

#endif /* __CUVNET_REGULARIZERS_HPP__ */
