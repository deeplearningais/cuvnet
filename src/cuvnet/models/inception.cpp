#include <cuvnet/ops.hpp>
#include "inception.hpp"

namespace cuvnet { namespace models {
    inception_layer::inception_layer(op_ptr in, const std::vector<boost::tuple<int,int,int> >& m)
        : m_input(in)
    {
        std::vector<op_ptr> stackable;
        std::vector<boost::tuple<int, int, int> >::const_iterator it;
        for(it = m.begin(); it != m.end(); it++){
            int fs = it->get<0>();
            int compress = it->get<1>();
            int n_dst = it->get<2>();

            determine_shapes(*in);
            int n_src_maps = in->result()->shape[0];

            op_ptr o;
            if(fs < 0){
                // max-pooling
                o = local_pool(m_input, compress, compress, cuv::alex_conv::PT_MAX);
            }else if(fs == 1){
                cuvAssert(n_src_maps > 1);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(m_input, n_dst, 
                            mlp_layer_opts().rectified_linear().weights_left()));
                register_submodel(*m_fclayers.back());
                o = m_fclayers.back()->m_output;
            }else{
                cuvAssert(n_src_maps > 1);
                m_fclayers.push_back(boost::make_shared<mlp_layer>(m_input, compress, 
                            mlp_layer_opts().weights_left()));
                register_submodel(*m_fclayers.back());
                m_convlayers.push_back(boost::make_shared<conv_layer>(m_input,
                            fs, n_dst, conv_layer_opts().rectified_linear().symmetric_padding()));
                register_submodel(*m_convlayers.back());
                o = m_convlayers.back()->m_output;
            }
            stackable.push_back(o);
        }
        m_output = concatenate(stackable, 0);
    }
}}
