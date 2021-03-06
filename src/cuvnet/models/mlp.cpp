#include <cuvnet/ops.hpp>
#include "initialization.hpp"
#include <cuvnet/op_utils.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/ops.hpp>
#include "mlp.hpp"

namespace{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("mlp"));
}

namespace cuvnet
{
    namespace models
    {
        void mlp_layer::reset_params(){
            if(m_weight_init_std < 0)
                initialize_dense_glorot_bengio(m_W, false);
            else{
                m_W->data() = 0.0f;
                cuv::add_rnd_normal(m_W->data(), m_weight_init_std);
            }
            if(m_bias){
                m_bias->data() = m_bias_default_value;
            }
        }

        mlp_layer::mlp_layer(mlp_layer::op_ptr X, unsigned int size, mlp_layer_opts args){
            determine_shapes(*X);
            if(args.m_weights_left)
                m_W    = input(cuv::extents[size][X->result()->shape[0]], args.m_group_name + "W");
            else
                m_W    = input(cuv::extents[X->result()->shape.back()][size], args.m_group_name + "W");
            m_W->set_learnrate_factor(args.m_learnrate_factor);

            boost::scoped_ptr<op_group> grp;
            if (!args.m_group_name.empty())
                grp.reset(new op_group(args.m_group_name, args.m_unique_group));

            if(args.m_want_dropout)
                X = m_noiser = zero_out(X, 0.5);

            if(args.m_want_bias){
                m_bias = input(cuv::extents[size], args.m_group_name + "b");
                m_bias->set_learnrate_factor(args.m_learnrate_factor_bias);

                if(args.m_weights_left)
                    m_linear_output = mat_plus_vec( prod(m_W, X), m_bias, 0);
                else
                    m_linear_output = mat_plus_vec( prod(X, m_W), m_bias, 1);
            }else{
                if(args.m_weights_left)
                    m_linear_output = prod(m_W, X);
                else
                    m_linear_output = prod(X, m_W);
            }

            unsigned int ndim = X->result()->shape.size();
            if(args.m_want_maxout)
                m_linear_output = tuplewise_op(m_linear_output, args.m_weights_left ? 0 : ndim-1, 
                        args.m_maxout_N, cuv::alex_conv::TO_MAX);

            if(args.m_nonlinearity)
                m_output = args.m_nonlinearity(m_linear_output);
            else
                m_output = m_linear_output;

            m_bias_default_value = args.m_bias_default_value;
            m_weight_init_std = args.m_weight_init_std;
            m_verbose = args.m_verbose;
            if(m_verbose){
                LOG4CXX_WARN(g_log, "#in: "<< m_W->data().shape(0) << ", #out: " << m_W->data().shape(1)
                        << ", #params: " 
                        << m_W->data().size() + (m_bias ? m_bias->data().size(): 0));
            }
        }
        void mlp_layer::set_predict_mode(bool b){
            if(m_noiser)
                m_noiser->set_active(!b);
        }
        void mlp_layer::register_watches(monitor& mon){
            if(!m_verbose)
                return;

            mon.add(monitor::WP_FULL_WEIGHT_STATS, m_W, m_W->name());

            mon.add(monitor::WP_SINK_ONCE_STATS,                                                                                                                                           
                    m_linear_output, m_W->name() + "_linout", 0);
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
                m_layers[i] = mlp_layer(o, hlsizes[i], mlp_layer_opts().rectified_linear());
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
