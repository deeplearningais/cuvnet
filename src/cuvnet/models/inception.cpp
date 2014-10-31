#include <cuvnet/ops.hpp>
#include "inception.hpp"

namespace cuvnet { namespace models {
    inception_layer::inception_layer(op_ptr in, const std::vector<boost::tuple<int,int,int> >& m)
        : m_input(in)
    {
        std::vector<op_ptr> stackable;
        std::vector<boost::tuple<int, int, int> >::const_iterator it;

        bool copy = true;
        op_ptr flat_input = flatten(m_input, 2, copy);

        for(it = m.begin(); it != m.end(); it++){
            int fs = it->get<0>();
            int compress = it->get<1>();
            int n_dst = it->get<2>();

            determine_shapes(*in);
            int n_src_maps  = in->result()->shape[0];
            int n_src_pix_y = in->result()->shape[1];
            int n_src_pix_x = in->result()->shape[2];
            //int n_batch     = in->result()->shape[3];

            op_ptr o;
            if(fs < 0){
                // max-pooling, then 1x1 convolution
                o = local_pool(m_input, compress, 1, cuv::alex_conv::PT_MAX);
                op_ptr flat_o = flatten(o, 2, copy);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(flat_o, n_dst, 
                            mlp_layer_opts().with_bias(false).weights_left()));
                register_submodel(*m_fclayers.back());
                o = reshape(m_fclayers.back()->m_output, cuv::extents[n_dst][n_src_pix_y][n_src_pix_x][-1], copy);
                //o = reshape(flat_o, cuv::extents[n_src_maps][n_src_pix_y][n_src_pix_x][-1], copy);
            }else if(fs == 1){
                cuvAssert(n_src_maps > 1);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(flat_input, n_dst, 
                            mlp_layer_opts().rectified_linear().weights_left()));
                register_submodel(*m_fclayers.back());
                o = reshape(m_fclayers.back()->m_output, cuv::extents[n_dst][n_src_pix_y][n_src_pix_x][-1], copy);
            }else{
                cuvAssert(n_src_maps > 1);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(flat_input, compress, 
                            mlp_layer_opts().weights_left()));
                register_submodel(*m_fclayers.back());
                o = reshape(m_fclayers.back()->m_output, cuv::extents[compress][n_src_pix_y][n_src_pix_x][-1], copy);
                m_convlayers.push_back(boost::make_shared<conv_layer>(o,
                            fs, n_dst, conv_layer_opts().rectified_linear().symmetric_padding(-1)));
                register_submodel(*m_convlayers.back());
                o = m_convlayers.back()->m_output;
            }
            stackable.push_back(o);
        }
        m_output = concatenate(stackable, 0);
    }
}}
