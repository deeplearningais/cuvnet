#include <cuvnet/ops.hpp>
#include "inception.hpp"

namespace cuvnet { namespace models {
    inception_layer::inception_layer(op_ptr in, const std::vector<boost::tuple<int,int,int> >& m, const std::string& label, bool copy, bool nonlinear)
        : m_input(in)
    {
        init(in, m, label, copy, nonlinear);
    }
    void
        inception_layer::init(op_ptr in, const std::vector<boost::tuple<int,int,int> >& m, const std::string& label, bool copy, bool nonlinear)
    {
        m_input = in;
        std::vector<op_ptr> stackable;
        std::vector<boost::tuple<int, int, int> >::const_iterator it;

        bool verbose = true;
        bool with_bias = true;
        float bias = 0.0f;

        for(it = m.begin(); it != m.end(); it++){
            int fs = it->get<0>();
            int compress = it->get<1>();
            int n_dst = it->get<2>();
            std::string lstr = "il"+label + "_" + boost::lexical_cast<std::string>(fs+1);

            determine_shapes(*m_input);
            int n_src_maps  = m_input->result()->shape[0];
            //int n_src_pix_y = m_input->result()->shape[1];
            //int n_src_pix_x = m_input->result()->shape[2];
            //int n_batch     = m_input->result()->shape[3];

            float wstd = 0.01f;
            if(fs == 3)
                wstd = 0.04;
            else if(fs == 5)
                wstd = 0.08;

            op_ptr o;
            if(fs < 0){
                // max-pooling, then 1x1 convolution
                o = pooling_cuDNN(m_input, cuv::alex_conv::PT_MAX, compress, compress, 1, 1);
                m_convlayers.push_back(boost::make_shared<conv_layer>(o, 1, n_dst, 
                            conv_layer_opts().with_bias(with_bias,bias).group(lstr, true).verbose(verbose)));
                o = m_convlayers.back()->m_output;
                register_submodel(*m_convlayers.back());
            }else if(fs == 0){
                // (intermediate) softmax-output type
                // 5x5/3 avg-pooling, 1x1 convolution -> compress, fc-layer -> n_dst
                o = pooling_cuDNN(m_input, cuv::alex_conv::PT_AVG, 5, 5, 3, 3);
                determine_shapes(*o);
                //int sx = o->result()->shape[1];

                m_convlayers.push_back(boost::make_shared<conv_layer>(o, 1, compress, 
                            conv_layer_opts().with_bias(with_bias,bias)
                            //.linear()
                            .rectified_linear(!copy)
                            .weight_default_std(0.01)
                            .group(lstr, true).verbose(verbose)));
                register_submodel(*m_convlayers.back());
                o = m_convlayers.back()->m_output;
                o = flatten(o, 2, copy);

                m_fclayers.push_back(boost::make_shared<mlp_layer>(o, n_dst, 
                            mlp_layer_opts().weight_init_std(0.01).with_bias(with_bias,bias).group(lstr, true).verbose(verbose)));

                register_submodel(*m_fclayers.back());

                o = m_fclayers.back()->m_output;
            }else if(fs == 1){
                cuvAssert(n_src_maps > 1);
                m_convlayers.push_back(boost::make_shared<conv_layer>(m_input, 1, n_dst, 
                            conv_layer_opts().with_bias(with_bias,bias).group(lstr, true).verbose(verbose)));
                o = m_convlayers.back()->m_output;
                register_submodel(*m_convlayers.back());
            }else{
                cuvAssert(n_src_maps > 1);
                m_convlayers.push_back(boost::make_shared<conv_layer>(m_input, 1, compress, 
                            conv_layer_opts().with_bias(with_bias,bias)
                            .rectified_linear(!copy)
                            //.linear()
                            .group(lstr,true).verbose(verbose)));
                register_submodel(*m_convlayers.back());
                m_convlayers.push_back(boost::make_shared<conv_layer>(m_convlayers.back()->m_output,
                            fs, n_dst, conv_layer_opts().with_bias(with_bias,bias).symmetric_padding(-1).group(lstr, true).verbose(verbose).use_cudnn().weight_default_std(wstd)));
                register_submodel(*m_convlayers.back());
                o = m_convlayers.back()->m_output;
            }
            stackable.push_back(o);
        }
        if(stackable.size() == 1)
            m_output = nonlinear ? rectified_linear(stackable.front(), false) : stackable.front();
        else
            m_output = nonlinear ? rectified_linear(concatenate(stackable, 1), false) : concatenate(stackable, 1);
    }
}}
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::inception_layer);
