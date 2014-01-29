#ifndef __CUVNET_MODELS_ALEXCONV_LAYERS_HPP__
#     define __CUVNET_MODELS_ALEXCONV_LAYERS_HPP__
#include <boost/function.hpp>
#include <cuvnet/models/models.hpp>

namespace cuvnet { namespace models {
    
    namespace detail {
        inline boost::shared_ptr<Op> no_op(boost::shared_ptr<Op> x){
            return x;
        }
    }

    struct alexconv_layer_maker{
        typedef boost::shared_ptr<Op> op_ptr;
        typedef boost::shared_ptr<ParameterInput> input_ptr;
        typedef boost::shared_ptr<Sink> sink_ptr;

        alexconv_layer_maker(op_ptr inp, unsigned int fs=5, unsigned int n_out=64)
            :m_input(inp)
            ,m_learnrate_factor(1.f)
            ,m_random_sparse(false)
            ,m_verbose(false)
            ,m_padding(-1)
            ,m_stride(1)
            ,m_n_groups(1)
            ,m_partial_sum(-1)
            ,m_want_pooling(false)
            ,m_pool_size(2)
            ,m_pool_stride(2)
            ,m_pool_type(cuv::alex_conv::PT_MAX)
            ,m_nonlinearity(detail::no_op)
            ,m_want_contrast_norm(false)
            ,m_want_response_normalization(false)
            ,m_rn_alpha(0.5)
            ,m_rn_beta(0.5)
            ,m_fs(fs)
            ,m_n_out(n_out)
            ,m_want_bias(false)
            ,m_scat_n_inputs(0)
            ,m_scat_J(0)
            ,m_scat_C(0)
            ,m_want_maxout(false)
            ,m_want_dropout(false)
        {}

        inline alexconv_layer_maker& rectified_linear(){
            m_nonlinearity = cuvnet::rectified_linear;
            return *this;
        }

        op_ptr m_input;

        float m_learnrate_factor;
        inline alexconv_layer_maker& learnrate_factor(float f){ m_learnrate_factor = f;  return *this;}

        bool m_random_sparse;
        inline alexconv_layer_maker& random_sparse(bool b=true){ m_random_sparse = b; return *this;}

        float m_verbose;
        inline alexconv_layer_maker& verbose(bool b){ m_verbose = b;  return *this;}

        int m_padding;
        unsigned int m_stride, m_n_groups;
        int m_partial_sum;

        bool m_want_pooling;
        unsigned int m_pool_size;
        unsigned int m_pool_stride;
        cuv::alex_conv::pool_type m_pool_type;

        inline alexconv_layer_maker& pool( 
                int pool_size, int pool_stride=-1, cuv::alex_conv::pool_type pt = cuv::alex_conv::PT_MAX){
            m_want_pooling = true;
            m_pool_size = pool_size;
            m_pool_stride = pool_stride < 0 ? pool_size : pool_stride;
            return *this;
        }

        boost::function<op_ptr(op_ptr)> m_nonlinearity;
        inline alexconv_layer_maker& nonlinearity(
                boost::function<op_ptr(op_ptr)> func){
            m_nonlinearity = func;
            return *this;
        };


        bool m_want_contrast_norm;
        /// subtract local mean?
        inline alexconv_layer_maker& contrast_norm(int n, float alpha=0.001, float beta=0.5){
            m_want_contrast_norm = true;
            m_rn_N = n;
            m_rn_alpha = alpha;
            m_rn_beta = beta;
            return *this;
        }
        bool m_want_response_normalization;
        int  m_rn_N;
        float m_rn_alpha, m_rn_beta;
        inline alexconv_layer_maker& response_norm(int n, float alpha=0.5f, float beta=0.5f)
        {
            m_want_response_normalization = true;
            m_rn_N = n;
            m_rn_alpha = alpha;
            m_rn_beta = beta;
            return *this;
        }

        /// negative padding will amount to a size-keeping convolution
        inline alexconv_layer_maker& padding(int i=-1){ m_padding = i; return *this;}
        inline alexconv_layer_maker& stride(unsigned int i){ m_stride = i; return *this;}
        inline alexconv_layer_maker& n_groups(unsigned int i){ m_n_groups = i; return *this;}
        inline alexconv_layer_maker& partial_sum(int i){ m_partial_sum = i; return *this;}

        unsigned int m_fs, m_n_out;
        inline alexconv_layer_maker& fs(unsigned int i){ m_fs = i; return *this;}
        inline alexconv_layer_maker& n_out(unsigned int i){ m_n_out = i; return *this;}

        bool m_want_bias;
        inline alexconv_layer_maker& with_bias(bool b=true){ m_want_bias = b; return *this; }

        int m_scat_n_inputs, m_scat_J, m_scat_C;
        inline alexconv_layer_maker& set_scattering_transform_params(int J, int C, int n_inputs=-1){
            m_scat_n_inputs = n_inputs;
            m_scat_J = J;
            m_scat_C = C;
            return *this;
        }

        bool m_want_maxout;
        int m_maxout_N;
        inline alexconv_layer_maker& maxout(int n){
            m_want_maxout = n > 1;
            m_maxout_N = n;
            return *this;
        }

        bool m_want_dropout;
        inline alexconv_layer_maker& dropout(bool b=true){
            m_want_dropout = b;
            return *this;
        }
    };
    struct alexconv_layer
    : public model {
        public:
            typedef boost::shared_ptr<Op> op_ptr;
            typedef boost::shared_ptr<ParameterInput> input_ptr;
            typedef boost::shared_ptr<Sink> sink_ptr;

            input_ptr m_weights;
            input_ptr m_bias;
            op_ptr m_input;

            boost::shared_ptr<Noiser> m_noiser;

            op_ptr m_output;
            bool m_verbose;

            float m_learnrate_factor;

            int m_scat_n_inputs, m_scat_J, m_scat_C;

            alexconv_layer(const alexconv_layer_maker& cfg);

            /// empty ctor for serialization
            alexconv_layer():m_verbose(false){};

            inline alexconv_layer& set_learnrate_factor(float f){
            	m_learnrate_factor = f;
            	return *this;
            }

            inline void set_noiser_active(bool b){
                if(m_noiser)
                    m_noiser->set_active(b);
            }
            void register_watches(monitor& mon);
            std::vector<Op*> get_params();
            void reset_params();

            virtual ~alexconv_layer();

        private:
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version){
                    ar & boost::serialization::base_object<model>(*this);
                    ar & m_weights & m_bias & m_output & m_verbose;
                    ar & m_scat_n_inputs & m_scat_J & m_scat_C;
                    ar & m_input;
                    if(version > 0)
                        ar & m_noiser;
                }
    };
} }

BOOST_CLASS_EXPORT_KEY(cuvnet::models::alexconv_layer)
BOOST_CLASS_VERSION(cuvnet::models::alexconv_layer, 1);

#endif /* __CUVNET_MODELS_ALEXCONV_LAYERS_HPP__ */
