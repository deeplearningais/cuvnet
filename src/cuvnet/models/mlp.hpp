#ifndef __CUVNET_MODELS_MLP_HPP__
#     define __CUVNET_MODELS_MLP_HPP__

#include <boost/function.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops.hpp>
#include "models.hpp"
#include "logistic_regression.hpp"

namespace cuvnet
{
    class monitor;
    namespace models
    {
        struct mlp_layer_opts{
            private:
                typedef boost::shared_ptr<Op> op_ptr;
                boost::function<op_ptr(op_ptr)> m_nonlinearity;
                bool m_weights_left;
                bool m_want_bias;
                float m_bias_default_value;
                bool m_want_maxout;
                int m_maxout_N;
                bool m_want_dropout;
                bool m_dropout_memopt;
                std::string m_group_name;
                bool m_unique_group;
                float m_learnrate_factor;
                float m_learnrate_factor_bias;
                bool m_verbose;
                float m_weight_init_std;
            public:
                friend class mlp_layer;
                /**
                 * ctor.
                 * @param X input of the mlp layer
                 * @param size number of neurons in the mlp layer
                 */
                mlp_layer_opts()
                    :m_weights_left(false)
                    ,m_want_bias(true)
                    ,m_bias_default_value(0.f)
                    ,m_want_maxout(false)
                    ,m_want_dropout(false)
                    ,m_dropout_memopt(false)
                    ,m_group_name("mlplayer")
                    ,m_unique_group(true)
                    ,m_learnrate_factor(1.f)
                    ,m_learnrate_factor_bias(1.f)
                    ,m_verbose(false)
                    ,m_weight_init_std(-1.f)
                {
                }
                /**
                 * set that weights are multiplied from left
                 * @param b verbosity
                 */
                inline mlp_layer_opts& weights_left(bool b=true){
                    m_weights_left = b;
                    return *this;
                }

                /**
                 * set verbosity (records mean, variance of weights and outputs after every epoch)
                 * @param b verbosity
                 */
                inline mlp_layer_opts& verbose(bool b=true){
                    m_verbose = b;
                    return *this;
                }

                /**
                 * copy the current parameters, useful if you want to have a default parameter set.
                 *
                 * @begincode
                 * const mlp_layer_opts def = mlp_layer_opts().verbose();
                 * mlp_layer l(inp, 10, def.copy().tanh());
                 * @endcode
                 */
                inline mlp_layer_opts copy()const{
                    return *this;
                }

                /**
                 * set rectified linear function as activation function and bias default to 1.
                 * @param mem_optimized if no-one else needs the result of the matrix product output, this is faster and more memory efficient.
                 */
                inline mlp_layer_opts& rectified_linear(bool mem_optimized=false){
                    m_nonlinearity = boost::bind(cuvnet::rectified_linear, _1, mem_optimized);
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
                 * @param b if true , use a bias after weight multiplication.
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
                 * @param mem_optimized if no-one else needs the result of the matrix product output, this is faster and more memory efficient.
                 */
                inline mlp_layer_opts& dropout(bool b=true, bool mem_optimized=false){
                    m_want_dropout = b;
                    m_dropout_memopt = mem_optimized;
                    return *this;
                }

                /**
                 * Set the learnrate factor for the weights.
                 * @param fW learnrate factor for weights
                 * @param fB learnrate factor for bias
                 */
                inline mlp_layer_opts& learnrate_factor(float fW, float fB=-1.f){ 
                    m_learnrate_factor = fW;
                    m_learnrate_factor_bias = (fB < 0) ? fW : fB;
                    return *this;
                }

                /**
                 * Specify group name to be used for all ops inside the layer.
                 * @param name the name for all mlp-layer ops
                 * @param unique if true, append a unique number to the name.
                 */
                inline mlp_layer_opts& group(std::string name="", bool unique=true) { m_group_name = name; return *this;  }

                /**
                 * Specify the standard deviation of the weight initialization.
                 *
                 * @param Negative values (default) means that the weights are initialized with the mechanism from Glorot & Bengio.
                 */
                inline mlp_layer_opts& weight_init_std(float f) { m_weight_init_std = f; return *this;  }
        };

        struct mlp_layer
        : public model{
            private:
                typedef boost::shared_ptr<ParameterInput> input_ptr;
                typedef boost::shared_ptr<Op> op_ptr;
            public:
                input_ptr m_W, m_bias;
                boost::shared_ptr<Noiser> m_noiser;
                op_ptr m_output, m_linear_output;
                float m_bias_default_value;
                float m_weight_init_std;
                bool m_verbose;
                /**
                 * ctor.
                 * @param X input to the hidden layer
                 * @param size size of the hidden layer.
                 */
                mlp_layer(op_ptr X, unsigned int size, mlp_layer_opts opts = mlp_layer_opts());
                mlp_layer(){} ///< default ctor for serialization
                virtual std::vector<Op*> get_params();
                virtual void register_watches(monitor& mon);
                virtual void reset_params();
                virtual void set_predict_mode(bool b=true);
                virtual ~mlp_layer(){}
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version) {
                        ar & boost::serialization::base_object<model>(*this);
                        ar & m_output & m_W & m_bias;
                        ar & m_bias_default_value;
                        if(version > 1)
                            ar & m_verbose;
                        if(version > 2)
                            ar & m_noiser;
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
                    mlp_classifier(){} ///< default ctor for serialization
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
BOOST_CLASS_VERSION(cuvnet::models::mlp_layer, 3);

#endif /* __CUVNET_MODELS_MLP_HPP__ */
