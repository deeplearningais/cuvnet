#include <cuvnet/ops.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/models/initialization.hpp>
#include "conv_layer.hpp"
#include <cuvnet/tools/monitor.hpp>
namespace{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("conv_layer"));
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
        ,m_shared_weight(false)
    {

        int padding;
        if(cfg.m_padding >= 0) padding = cfg.m_padding;
        else                   padding = fs / 2;

        determine_shapes(*m_input);
        unsigned int n_srcmaps = m_input->result()->shape[0];
        unsigned int n_fltpix  = fs * fs;

        m_bias_default_value = cfg.m_bias_default_value;
        m_weight_default_std = cfg.m_weight_default_std;

        cuvAssert(n_srcmaps % cfg.m_n_groups == 0);
        cuvAssert((n_srcmaps < 4) || (n_srcmaps % 4 == 0));
        int n_filter_channels = n_srcmaps / cfg.m_n_groups;
        if(cfg.m_random_sparse && cfg.m_n_filter_channels > 0)
            n_filter_channels = cfg.m_n_filter_channels;
        {     
        boost::scoped_ptr<op_group> grp;
         if (cfg.m_group_name_wb!="")
            grp.reset(new op_group(cfg.m_group_name_wb));

        if (cfg.m_shared_weight)
        {
            m_shared_weight = true;
            determine_shapes(*cfg.m_shared_weight);
            
            cuvAssert(cfg.m_shared_weight->result()->shape[0] == n_filter_channels);
            cuvAssert(cfg.m_shared_weight->result()->shape[1] == n_fltpix);
            cuvAssert(cfg.m_shared_weight->result()->shape[2] == n_out);
            m_weights = cfg.m_shared_weight;
        }
        else
        {
            m_weights = input(cuv::extents[n_filter_channels][n_fltpix][n_out], cfg.m_group_name + "W" + cfg.m_varname_suffix);
        }

        if(cfg.m_want_bias){
            if (cfg.m_shared_bias)
                m_bias = cfg.m_shared_bias;
            else
                m_bias    = input(cuv::extents[n_out], cfg.m_group_name + "b" + cfg.m_varname_suffix);
            }
        }

        boost::scoped_ptr<op_group> grp;
        if (!cfg.m_group_name.empty())
            grp.reset(new op_group(cfg.m_group_name, cfg.m_unique_group));

        inp = m_input;
        if(cfg.m_want_dropout){
            inp = m_noiser = zero_out(inp, cfg.m_dropout_rate);
        }
        auto conv = convolve(inp, m_weights, padding>=0, padding, cfg.m_stride, cfg.m_n_groups, 0);
        if(cfg.m_symmetric_padding)
            conv->set_symmetric_padding(true);

        determine_shapes(*conv);

        LOG4CXX_WARN(g_log, "#srcmaps: "<<n_srcmaps << ", #out: "<<n_out 
                << ", #params: " << m_weights->data().size() + (m_bias ? m_bias->data().size(): 0)
                << ", #neuron: " << conv->result()->shape[1] * conv->result()->shape[2] * n_out
                );

        int partial_sum = cfg.m_partial_sum;
        // determine partial_sum automatically if not given
        if (cfg.m_partial_sum < 0){
            int conv_shape_y = conv->result()->shape[1];
            int conv_shape_x = conv->result()->shape[2];
            int num_modules = conv_shape_y * conv_shape_x;
            //partial_sum = num_modules;
            //while((partial_sum/2)*2 == partial_sum && partial_sum > 4)
            //    partial_sum /= 2;


            partial_sum = 1;
            for(int ps=1; ps < 128; ps ++){
                if(num_modules % ps == 0      // divisibla
                        && num_modules / ps >= 128)  // enough work left to spawn many threads
                    partial_sum = ps;
            }
            LOG4CXX_WARN(g_log, "Automatically determined partial_sum: " <<partial_sum);
        }else if(cfg.m_partial_sum > 0){
            int conv_shape_y = conv->result()->shape[1];
            int conv_shape_x = conv->result()->shape[2];
            int num_modules = conv_shape_y * conv_shape_x;
            if(num_modules % partial_sum != 0){
                LOG4CXX_FATAL(g_log, "Given partial_sum: " <<partial_sum << " does not divide num_modules "<<num_modules);
            }else{
                LOG4CXX_WARN(g_log, "Supplied partial_sum: " <<partial_sum);
            }
        }else{
                LOG4CXX_WARN(g_log, "Supplied partial_sum: " <<partial_sum);
        }
        conv->set_partial_sum(partial_sum);
        if(cfg.m_random_sparse)
            conv->set_random_sparse(cfg.m_n_filter_channels);
        m_output = conv;
        if(cfg.m_want_bias){
            m_output = mat_plus_vec(conv, m_bias, 0);
        }
        m_linear_output = m_output;

        if(cfg.m_want_maxout){
            m_output = tuplewise_op(m_output, 0, cfg.m_maxout_N, cuv::alex_conv::TO_MAX);
        }

        // finally: apply the non-linearity.
        if(cfg.m_nonlinearity)
            m_output = cfg.m_nonlinearity(m_output);

        // Caffe: normalization /after/ ReLU
        // https://github.com/BVLC/caffe/blob/master/examples/imagenet/alexnet_train.prototxt
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
        
        // Krizhevsky et al.: pooling /after/ response normalization
        m_output_before_pooling = m_output;
        if(cfg.m_want_pooling){
            m_output = local_pool(m_output, cfg.m_pool_size, cfg.m_pool_stride, cfg.m_pool_type);
        }

    }


    void conv_layer::register_watches(monitor& mon){
        if(!m_verbose)
            return;
         
        if (!m_shared_weight)
        {
        mon.add(monitor::WP_CONV_WEIGHT_STATS, m_weights, m_weights->name());

        mon.add(monitor::WP_SINK_ONCE_STATS,
                m_linear_output, m_weights->name() + "_linout", 0);
        }
    }

    std::vector<Op*>
    conv_layer::get_params(){
        std::vector<Op*> params;
        params.push_back(m_weights.get());
        if(m_bias)
            params.push_back(m_bias.get());
        return params;
    }

    void conv_layer::set_predict_mode(bool b){
        if(m_noiser)
            m_noiser->set_active(!b);
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
