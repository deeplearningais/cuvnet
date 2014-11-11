#include <cuvnet/ops.hpp>
#include "inception.hpp"

namespace cuvnet { namespace models {
    inception_layer::inception_layer(op_ptr in, const std::vector<boost::tuple<int,int,int> >& m, const std::string& label, bool copy)
        : m_input(in)
    {
        init(in, m, label, copy);
    }
    void
        inception_layer::init(op_ptr in, const std::vector<boost::tuple<int,int,int> >& m, const std::string& label, bool copy)
    {
        std::vector<op_ptr> stackable;
        std::vector<boost::tuple<int, int, int> >::const_iterator it;

        bool verbose = true;
        bool with_bias = true;
        float bias = 0.1f;
        op_ptr flat_input = flatten(m_input, 2, copy);

        for(it = m.begin(); it != m.end(); it++){
            int fs = it->get<0>();
            int compress = it->get<1>();
            int n_dst = it->get<2>();
            std::string lstr = "il"+label + "_" + boost::lexical_cast<std::string>(fs+1);

            determine_shapes(*m_input);
            int n_src_maps  = m_input->result()->shape[0];
            int n_src_pix_y = m_input->result()->shape[1];
            int n_src_pix_x = m_input->result()->shape[2];
            //int n_batch     = m_input->result()->shape[3];

            op_ptr o;
            if(fs < 0){
                // max-pooling, then 1x1 convolution
                o = local_pool(m_input, compress, 1, cuv::alex_conv::PT_MAX);
                op_ptr flat_o = flatten(o, 2, copy);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(flat_o, n_dst, 
                            mlp_layer_opts().with_bias(with_bias,bias).weights_left().rectified_linear(!copy).group(lstr, true).verbose(verbose)));
                register_submodel(*m_fclayers.back());
                o = reshape(m_fclayers.back()->m_output, cuv::extents[n_dst][n_src_pix_y][n_src_pix_x][-1], copy);
                //o = reshape(flat_o, cuv::extents[n_src_maps][n_src_pix_y][n_src_pix_x][-1], copy);
            }else if(fs == 0){
                // (intermediate) softmax-output type
                // 5x5/3 avg-pooling, 1x1 convolution -> compress, fc-layer -> n_dst
                o = local_pool(m_input, 5, 3, cuv::alex_conv::PT_AVG);
                determine_shapes(*o);
                int sx = o->result()->shape[1];

                op_ptr flat_o = flatten(o, 2, copy);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(flat_o, compress, 
                            mlp_layer_opts().with_bias(with_bias,bias).weights_left().rectified_linear(!copy).group(lstr, true).verbose(verbose)));
                register_submodel(*m_fclayers.back());
                o = m_fclayers.back()->m_output;
                o = reshape(o, cuv::extents[compress][sx][sx][-1], copy);
                o = reorder_from_conv(o);
                o = flatten(o, 2);

                m_fclayers.push_back(boost::make_shared<mlp_layer>(o, n_dst, 
                            mlp_layer_opts().with_bias(with_bias,bias).rectified_linear(!copy).group(lstr, true).verbose(verbose)));
                register_submodel(*m_fclayers.back());
                o = m_fclayers.back()->m_output;
            }else if(fs == 1){
                cuvAssert(n_src_maps > 1);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(flat_input, n_dst, 
                            mlp_layer_opts().with_bias(with_bias,bias).rectified_linear(!copy).weights_left().group(lstr, true).verbose(verbose)));
                register_submodel(*m_fclayers.back());
                o = reshape(m_fclayers.back()->m_output, cuv::extents[n_dst][n_src_pix_y][n_src_pix_x][-1], copy);
            }else{
                cuvAssert(n_src_maps > 1);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(flat_input, compress, 
                            mlp_layer_opts().with_bias(with_bias,bias).weights_left().rectified_linear(!copy).group(lstr,true).verbose(verbose)));
                register_submodel(*m_fclayers.back());
                o = reshape(m_fclayers.back()->m_output, cuv::extents[compress][n_src_pix_y][n_src_pix_x][-1], copy);
                m_convlayers.push_back(boost::make_shared<conv_layer>(o,
                            fs, n_dst, conv_layer_opts().with_bias(with_bias,bias).rectified_linear(!copy).symmetric_padding(-1).group(lstr, true).verbose(verbose)));
                register_submodel(*m_convlayers.back());
                o = m_convlayers.back()->m_output;
            }
            stackable.push_back(o);
        }
        if(stackable.size() == 1)
            m_output = stackable.front();
        else
            m_output = concatenate(stackable, 0);
    }
}}
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::inception_layer);
