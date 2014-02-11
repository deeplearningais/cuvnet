#ifndef __CUVNET_MODELS_MLP_HPP__
#     define __CUVNET_MODELS_MLP_HPP__

#include <boost/function.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops.hpp>
#include "models.hpp"
#include "logistic_regression.hpp"

namespace cuvnet
{
    namespace models
    {
        struct mlp_layer_opts{
            private:
                typedef boost::shared_ptr<Op> op_ptr;
                boost::function<op_ptr(op_ptr)> m_nonlinearity;
                bool m_want_bias;
                float m_bias_default_value;
                bool m_want_maxout;
                int m_maxout_N;
                bool m_want_dropout;
                std::string m_group_name;
                bool m_unique_group;
            public:
                friend class mlp_layer;
                /**
                 * ctor.
                 * @param X input of the mlp layer
                 * @param size number of neurons in the mlp layer
                 */
                mlp_layer_opts()
                    :m_want_bias(true)
                    ,m_bias_default_value(0.f)
                    ,m_want_maxout(false)
                    ,m_want_dropout(false)
                    ,m_group_name("mlplayer")
                    ,m_unique_group(true)
                {
                }

                /**
                 * set rectified linear function as activation function and bias default to 1.
                 */
                inline mlp_layer_opts& rectified_linear(){
                    m_nonlinearity = cuvnet::rectified_linear;
                    m_bias_default_value = 1.f;
                    return *this;
                }

                /**
                 * set tanh function as activation function
                 */
                inline mlp_layer_opts& tanh(){
                    m_nonlinearity = cuvnet::tanh;
                    return *this;
                }

                /**
                 * set logistic function as activation function
                 */
                inline mlp_layer_opts& logistic(){
                    m_nonlinearity = (op_ptr (*)(op_ptr))cuvnet::logistic;
                    return *this;
                }

                /**
                 * Set the non-linearity.
                 * @param func a function that takes an Op and returns an op transformed by the non-linearity.
                 */
                mlp_layer_opts& non_linearity(boost::function<op_ptr(op_ptr)> f){
                    m_nonlinearity = f;
                    return *this;
                }

                /**
                 * Request/disable bias.
                 * @param b if true , use a bias after convolution.
                 */
                inline mlp_layer_opts& with_bias(bool b=true, float defaultval=0.f){ 
                    m_want_bias = b; 
                    m_bias_default_value = defaultval;
                    return *this; 
                }

                /**
                 * Use maxout for this layer.
                 * @param n the number of maps to take maximum over (non-overlapping), n==1 turns the feature off.
                 */
                inline mlp_layer_opts& maxout(int n){
                    m_want_maxout = n > 1;
                    m_maxout_N = n;
                    return *this;
                }

                /**
                 * Use dropout after pooling.
                 * @param b if true, use dropout.
                 */
                inline mlp_layer_opts& dropout(bool b=true){
                    m_want_dropout = b;
                    return *this;
                }

                /**
                 * Specify group name to be used for all ops inside the layer.
                 * @param name the name for all conv-layer ops
                 * @param unique if true, append a unique number to the name.
                 */
                inline mlp_layer_opts& group(std::string name="", bool unique=true) { m_group_name = name; return *this;  }
        };

        struct mlp_layer
        : public model{
            private:
                typedef boost::shared_ptr<ParameterInput> input_ptr;
                typedef boost::shared_ptr<Op> op_ptr;
            public:
                input_ptr m_W, m_bias;
                op_ptr m_output, m_linear_output;
                float m_bias_default_value;
                /**
                 * ctor.
                 * @param X input to the hidden layer
                 * @param size size of the hidden layer.
                 */
                mlp_layer(op_ptr X, unsigned int size, mlp_layer_opts opts = mlp_layer_opts());
                mlp_layer(){} ///< default ctor for serialization
                virtual std::vector<Op*> get_params();
                virtual void reset_params();
                virtual ~mlp_layer(){}
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version) {
                        ar & boost::serialization::base_object<model>(*this);
                        ar & m_output & m_W & m_bias;
                        ar & m_bias_default_value;
                    };
        };


        struct mlp_classifier
            : public metamodel<model>{
                private:
                    typedef boost::shared_ptr<ParameterInput> input_ptr;
                    typedef boost::shared_ptr<Op> op_ptr;
                    std::vector<mlp_layer> m_layers;
                    logistic_regression m_logreg;
                public:
                    mlp_classifier(input_ptr X, input_ptr Y, std::vector<unsigned int> hlsizes);
                    virtual ~mlp_classifier(){}
                private:
                    friend class boost::serialization::access;
                    template<class Archive>
                        void serialize(Archive& ar, const unsigned int version) {
                            ar & boost::serialization::base_object<metamodel<model> >(*this);;
                            ar & m_layers & m_logreg;
                        };
            };
    }
}
BOOST_CLASS_EXPORT_KEY(cuvnet::models::mlp_layer);
BOOST_CLASS_EXPORT_KEY(cuvnet::models::mlp_classifier);

#endif /* __CUVNET_MODELS_MLP_HPP__ */
