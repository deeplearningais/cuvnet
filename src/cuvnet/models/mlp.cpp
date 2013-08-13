#include <cuvnet/ops.hpp>
#include "initialization.hpp"
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include "mlp.hpp"

namespace cuvnet
{
    namespace models
    {
        void mlp_layer::reset_params(){
            initialize_dense_glorot_bengio(m_W, true);
        }

        mlp_layer::mlp_layer(mlp_layer::op_ptr X, unsigned int size){
            determine_shapes(*X);
            m_W    = input(cuv::extents[X->result()->shape[1]][size]);
            m_bias = input(cuv::extents[size]);

            m_output = tanh(
                    mat_plus_vec(
                        prod(X, m_W), m_bias, 1));
        }
        std::vector<Op*> 
        mlp_layer::get_params(){
            std::vector<Op*> params(2);
            params[0] = m_W.get();
            params[1] = m_bias.get();
            return params;
        }

        mlp_classifier::mlp_classifier(
                mlp_classifier::input_ptr X, 
                mlp_classifier::input_ptr Y, 
                std::vector<unsigned int> hlsizes){
            m_layers.resize(hlsizes.size());
            op_ptr o = X;
            for(unsigned int i=0; i< hlsizes.size(); i++) {
                m_layers[i] = mlp_layer(o, hlsizes[i]);
                register_submodel(m_layers[i]);
                o = m_layers[i].m_output;
            }
            m_logreg = logistic_regression(o, Y);
            register_submodel(m_logreg);
        }
    }
}
