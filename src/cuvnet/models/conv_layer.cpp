#include <cuvnet/ops.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/models/initialization.hpp>
#include "conv_layer.hpp"

namespace{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("layers"));
}

namespace cuvnet { namespace models {

    typedef boost::shared_ptr<Op> op_ptr;
    typedef boost::shared_ptr<ParameterInput> input_ptr;

    conv_layer::conv_layer(op_ptr inp, int fs, int n_out, const conv_layer_opts& cfg)
        :m_input(inp)
        ,m_verbose(cfg.m_verbose)
        ,m_learnrate_factor(cfg.m_learnrate_factor)
        ,m_learnrate_factor_bias(cfg.m_learnrate_factor_bias)
        ,m_scat_n_inputs(cfg.m_scat_n_inputs)
        ,m_scat_J(cfg.m_scat_J)
        ,m_scat_C(cfg.m_scat_C)
    {

        int padding;
        if(cfg.m_padding >= 0) padding = cfg.m_padding;
        else                   padding = fs / 2;

        determine_shapes(*m_input);
        unsigned int n_srcmaps = m_input->result()->shape[0];
        unsigned int n_fltpix  = fs * fs;
        LOG4CXX_WARN(g_log, "n_srcmaps: "<<n_srcmaps << ", n_out: "<<n_out);

        m_bias_default_value = cfg.m_bias_default_value;
        m_weight_default_std = cfg.m_weight_default_std;

        m_weights = input(cuv::extents[n_srcmaps / cfg.m_n_groups][n_fltpix][n_out], cfg.m_group_name + "W" + cfg.m_varname_suffix);
        if(cfg.m_want_bias){
            m_bias    = input(cuv::extents[n_out], cfg.m_group_name + "b" + cfg.m_varname_suffix);
        }

        boost::scoped_ptr<op_group> grp;
        if (!cfg.m_group_name.empty())
            grp.reset(new op_group(cfg.m_group_name, cfg.m_unique_group));

        auto ret = convolve(m_input, m_weights, padding>=0, padding, cfg.m_stride, cfg.m_n_groups, 0);
        determine_shapes(*ret);
        int partial_sum = cfg.m_partial_sum;
        // determine partial_sum automatically if not given
        if (cfg.m_partial_sum <= 0){
            int ret_shape_y = ret->result()->shape[1];
            int ret_shape_x = ret->result()->shape[2];
            int num_modules = ret_shape_y * ret_shape_x;
            partial_sum = num_modules;

            while((partial_sum/2)*2 == partial_sum && partial_sum > 4)
                partial_sum /= 2;
            //partial_sum = 4;
            LOG4CXX_WARN(g_log, "Automatically determined partial_sum: " <<partial_sum);
        }else{
            int ret_shape_y = ret->result()->shape[1];
            int ret_shape_x = ret->result()->shape[2];
            int num_modules = ret_shape_y * ret_shape_x;
            if(num_modules % partial_sum != 0){
                LOG4CXX_FATAL(g_log, "Given partial_sum: " <<partial_sum << " does not divide num_modules "<<num_modules);
            }else{
                LOG4CXX_WARN(g_log, "Supplied partial_sum: " <<partial_sum);
            }
        }
        boost::dynamic_pointer_cast<Convolve>(ret)->set_partial_sum(partial_sum);
        if(cfg.m_random_sparse)
            boost::dynamic_pointer_cast<Convolve>(ret)->set_random_sparse();
        if(cfg.m_want_bias){
            ret = mat_plus_vec(ret, m_bias, 0);
        }
        m_output = ret;
        m_linear_output = m_output;
        if(cfg.m_want_dropout){
            m_output = m_noiser = zero_out(m_output, 0.5);
        }
        if(cfg.m_want_maxout){
            m_output = tuplewise_op(m_output, 0, cfg.m_maxout_N, cuv::alex_conv::TO_MAX);
        }

        m_output_before_pooling = m_output;
        if(cfg.m_want_pooling){
            m_output = local_pool(m_output, cfg.m_pool_size, cfg.m_pool_stride, cfg.m_pool_type);
        }

        if(cfg.m_nonlinearity)
            m_output = cfg.m_nonlinearity(m_output);

        if(cfg.m_want_contrast_norm){
            cuv::tensor<float, cuv::host_memory_space> kernel(5);
            kernel[0] = 1.f;
            kernel[1] = 4.f;
            kernel[2] = 6.f;
            kernel[3] = 4.f;
            kernel[4] = 1.f;
            kernel /= cuv::sum(kernel);
            op_ptr mean = separable_filter(m_output, kernel);
            m_output = contrast_normalization(m_output - mean, cfg.m_rn_N, cfg.m_rn_alpha, cfg.m_rn_beta);
        }
        else if(cfg.m_want_response_normalization){
            m_output = response_normalization_cross_maps(m_output,
                    cfg.m_rn_N, cfg.m_rn_alpha, cfg.m_rn_beta);
        }
    }


    void conv_layer::register_watches(monitor& mon){
        if(!m_verbose)
            return;

        mon.add(monitor::WP_CONV_WEIGHT_STATS, m_weights, m_weights->name());

        mon.add(monitor::WP_SINK_ONCE_STATS,
                m_linear_output, m_weights->name() + "_linout", 0);
    }

    std::vector<Op*>
    conv_layer::get_params(){
        std::vector<Op*> params;
        params.push_back(m_weights.get());
        if(m_bias)
            params.push_back(m_bias.get());
        return params;
    }

    void conv_layer::reset_params(){
    	m_weights->set_learnrate_factor(m_learnrate_factor);
        if(m_bias)
            m_bias->set_learnrate_factor(m_learnrate_factor_bias);

        //initialize_alexconv_glorot_bengio(m_weights, 1, true);
        m_weights->data() = 0.f;
        cuv::add_rnd_normal(m_weights->data(), m_weight_default_std);
        //cuv::fill_rnd_uniform(m_weights->data());
        //m_weights->data() *= 0.02f;
        //m_weights->data() -= 0.01f;
        //int n_scat_inp = m_scat_n_inputs;
//        if(m_scat_J != 0){
//        	if(n_scat_inp < 0)
//        		n_scat_inp = m_weights->data().shape(0); // work on all channels of lower layer
//
//        	bool full = n_scat_inp == 3 || n_scat_inp == 1;
//            //bool neg = !full;
//            bool neg = false;
//
//        	initialize_alexconv_scattering_transform(
//        			boost::format("../src/filter_banks/psi_filters-%d-gabor-00-s2_x_t4_asqrt2.bin"),
//        			m_weights, n_scat_inp, m_scat_J, m_scat_C, full, false, neg);
//        }

        //m_weights->data() *= 0.3f;
        //m_weights->data() = 0.f;
        //for(unsigned int i=0; i<m_weights->data().shape(0); i++){
        //    for (int j = 0; j < m_weights->data().shape(2); ++j){
        //        m_weights->data()(i,m_weights->data().shape(1)/2,j) = 1.f;
        //    }
        //}
        if(m_bias){
            m_bias->data() = m_bias_default_value;
            m_bias->set_weight_decay_factor(0.f);
        }
    }

    conv_layer::~conv_layer(){
    }
} }
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::conv_layer)
