#include <cuvnet/ops.hpp>
#include "initialization.hpp"
#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/ops.hpp>
#include "mlp.hpp"

namespace cuvnet
{
    namespace models
    {
        void mlp_layer::reset_params(){
            initialize_dense_glorot_bengio(m_W, true);
            //m_W->data() = 0.0f;
            //cuv::add_rnd_normal(m_W->data(), 0.001f);
            if(m_bias){
                m_bias->data() = m_bias_default_value;
            }
        }

        mlp_layer::mlp_layer(mlp_layer::op_ptr X, unsigned int size, mlp_layer_opts args){
            determine_shapes(*X);
            m_W    = input(cuv::extents[X->result()->shape[1]][size], args.m_group_name + "W");
            m_W->set_learnrate_factor(args.m_learnrate_factor);

            boost::scoped_ptr<op_group> grp;
            if (!args.m_group_name.empty())
                grp.reset(new op_group(args.m_group_name, args.m_unique_group));

            if(args.m_want_bias){
                m_bias = input(cuv::extents[size], args.m_group_name + "b");
                m_bias->set_learnrate_factor(args.m_learnrate_factor_bias);
                m_linear_output = mat_plus_vec(
                        prod(X, m_W), m_bias, 1);
            }else
                m_linear_output = prod(X, m_W);

            if(args.m_want_dropout)
                m_linear_output = zero_out(m_linear_output, 0.5);

            if(args.m_want_maxout)
                m_linear_output = tuplewise_op(m_linear_output, 1, 
                        args.m_maxout_N, cuv::alex_conv::TO_MAX);

            if(args.m_nonlinearity)
                m_output = args.m_nonlinearity(m_linear_output);
            else
                m_output = m_linear_output;

            m_bias_default_value = args.m_bias_default_value;
            m_verbose = args.m_verbose;
        }
        void mlp_layer::register_watches(monitor& mon){
            if(!m_verbose)
                return;

            op_ptr m = mean(m_W);
            op_ptr v = mean(square(m_W)) - square(m);
            mon.add(monitor::WP_SCALAR_EPOCH_STATS, v, m_W->name() + "_var");
            mon.add(monitor::WP_SCALAR_EPOCH_STATS, m, m_W->name() + "_mean");

            op_ptr outvar = mean(var_to_vec(m_linear_output, 1));
            mon.add(monitor::WP_SCALAR_EPOCH_STATS,                                                                                                                                           
                    outvar, m_W->name() + "_linout_var", 0);

            op_ptr outmean = mean(mean_to_vec(m_linear_output, 1));
            mon.add(monitor::WP_SCALAR_EPOCH_STATS,                                                                                                                                           
                    outmean, m_W->name() + "_linout_mean", 0);
        }
        std::vector<Op*> 
        mlp_layer::get_params(){
            std::vector<Op*> params;
            params.push_back(m_W.get());
            if(m_bias)
                params.push_back(m_bias.get());
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
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::mlp_layer);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::mlp_classifier);
