#ifndef __CUVNET_MODELS_CONV_LAYERS_HPP__
#     define __CUVNET_MODELS_CONV_LAYERS_HPP__
#include <boost/function.hpp>
#include <cuvnet/models/models.hpp>

namespace cuvnet { namespace models {
    
    /**
     * Options for creating a convolution layer.
     *
     * This is an instance of the "named parameter idiom":
     * - there are quite a lot of default parameters
     * - all deviations can be specified using setters/getters
     * - the conv_layer class is not cluttered with
     *   variables only needed during construction
     */
    struct conv_layer_opts{
        typedef boost::shared_ptr<Op> op_ptr;
        typedef boost::shared_ptr<ParameterInput> input_ptr;
        typedef boost::shared_ptr<Sink> sink_ptr;

        /**
         * ctor.
         *
         * Sets all default values, only real mandatory argument is the input
         * to the convolution layer, which results in a 5x5 filter, 64
         * output-map conv_layer with full map-to-map connectivity and no
         * further gimmicks.
         */
        conv_layer_opts(op_ptr inp, unsigned int fs=5, unsigned int n_out=64)
            :m_input(inp)
            ,m_learnrate_factor(1.f)
            ,m_learnrate_factor_bias(1.f)
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
            ,m_want_contrast_norm(false)
            ,m_want_response_normalization(false)
            ,m_rn_alpha(0.5)
            ,m_rn_beta(0.5)
            ,m_fs(fs)
            ,m_n_out(n_out)
            ,m_want_bias(false)
            ,m_bias_default_value(0.f)
            ,m_weight_default_std(0.015f)
            ,m_scat_n_inputs(0)
            ,m_scat_J(0)
            ,m_scat_C(0)
            ,m_want_maxout(false)
            ,m_want_dropout(false)
            ,m_group_name("convlayer")
            ,m_unique_group(true)
        {}

        friend class conv_layer;
        private:
            op_ptr m_input;
            float m_learnrate_factor, m_learnrate_factor_bias;
            bool m_random_sparse;
            float m_verbose;
            int m_padding;
            unsigned int m_stride, m_n_groups;
            int m_partial_sum;
            bool m_want_pooling;
            unsigned int m_pool_size;
            unsigned int m_pool_stride;
            cuv::alex_conv::pool_type m_pool_type;
            boost::function<op_ptr(op_ptr)> m_nonlinearity;
            bool m_want_contrast_norm;
            bool m_want_response_normalization;
            int  m_rn_N;
            float m_rn_alpha, m_rn_beta;
            unsigned int m_fs, m_n_out;
            bool m_want_bias;
            float m_bias_default_value;
            float m_weight_default_std;
            int m_scat_n_inputs, m_scat_J, m_scat_C;
            bool m_want_maxout;
            int m_maxout_N;
            bool m_want_dropout;
            std::string m_group_name;
            bool m_unique_group;
        public:

        /**
         * set rectified linear function as activation function and bias default value to 1
         */
        inline conv_layer_opts& rectified_linear(){
            m_nonlinearity = cuvnet::rectified_linear;
            m_bias_default_value = 1.f;
            return *this;
        }


        /**
         * Set the learnrate factor for the weights.
         * @param fW learnrate factor for convolutional weights
         * @param fB learnrate factor for bias
         */
        inline conv_layer_opts& learnrate_factor(float fW, float fB=-1.f){ 
            m_learnrate_factor = fW;
            m_learnrate_factor_bias = (fB < 0) ? fW : fB;
            return *this;
        }

        /**
         * Set random sparse connectivity per group.
         *
         * There must be more than one group for this to work!
         */
        inline conv_layer_opts& random_sparse(bool b=true){ m_random_sparse = b; return *this;}

        /**
         * Set the layer to be verbose.
         *
         * Verbose layers record their mean and variance.
         */
        inline conv_layer_opts& verbose(bool b=true){ m_verbose = b;  return *this;}

        /**
         * Set pooling options.
         * By default no pooling is performed. 
         * @param pool_size how large the (square) area is over which is pooled
         * @param pool_stride stride between pooling areas (-1: equal to pool_size)
         * @param pt the pooling type (SUM, AVG, MAX)
         */
        inline conv_layer_opts& pool( 
                int pool_size, int pool_stride=-1, cuv::alex_conv::pool_type pt = cuv::alex_conv::PT_MAX){
            m_want_pooling = true;
            m_pool_size = pool_size;
            m_pool_stride = pool_stride < 0 ? pool_size : pool_stride;
            return *this;
        }

        /**
         * Set the non-linearity.
         * @param func a function that takes an Op and returns an op transformed by the non-linearity.
         */
        inline conv_layer_opts& nonlinearity(
                boost::function<op_ptr(op_ptr)> func){
            m_nonlinearity = func;
            return *this;
        };


        /**
         * use contrast normalization (after pooling, if specified).
         *
         * Uses subtractive and divisive normalization.
         * 
         * @see http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_contrast_normalization_layer
         */
        inline conv_layer_opts& contrast_norm(int n, float alpha=0.001, float beta=0.5){
            m_want_contrast_norm = true;
            m_rn_N = n;
            m_rn_alpha = alpha;
            m_rn_beta = beta;
            return *this;
        }

        /**
         * Use response normalization (after pooling, if specified).
         *
         * This is a normalization across maps.
         *
         * @see http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_%28across_maps%29
         */
        inline conv_layer_opts& response_norm(int n, float alpha=0.5f, float beta=0.5f)
        {
            m_want_response_normalization = true;
            m_rn_N = n;
            m_rn_alpha = alpha;
            m_rn_beta = beta;
            return *this;
        }

        // negative padding will amount to a size-keeping convolution
        /**
         * add padding around convolution.
         *
         * @param i padding amount: if negative, use size-keeping convolution
         */
        inline conv_layer_opts& padding(int i=-1){ m_padding = i; return *this;}

        /**
         * Increase stride of convolution.
         * @param i new stride
         */
        inline conv_layer_opts& stride(unsigned int i){ m_stride = i; return *this;}

        /**
         * Set the number of output groups.
         *
         * This can be used with or without random_sparse connections.
         *
         * @see random_sparse
         */
        inline conv_layer_opts& n_groups(unsigned int i){ m_n_groups = i; return *this;}

        /**
         * This is a speed/size tradeoff for gradient computation.
         */
        inline conv_layer_opts& partial_sum(int i){ m_partial_sum = i; return *this;}

        /**
         * Specify the filter size of the convolution.
         * @note that this can also be done in the constructor.
         * @param i edge length of filters will be (i times i)
         */
        inline conv_layer_opts& fs(unsigned int i){ m_fs = i; return *this;}

        /**
         * Specify number of output maps.
         * @note that this can also be done in the constructor.
         * @param i number of output maps
         */
        inline conv_layer_opts& n_out(unsigned int i){ m_n_out = i; return *this;}

        /**
         * Request/disable bias.
         * @param b if true , use a bias after convolution.
         */
        inline conv_layer_opts& with_bias(bool b=true, float defaultvalue=0.f){ 
            m_want_bias = b; 
            m_bias_default_value = defaultvalue;
            return *this; 
        }

        /**
         * Set parameters for scattering transform.
         * @deprecated this will/should be moved in a subclass.
         */
        inline conv_layer_opts& set_scattering_transform_params(int J, int C, int n_inputs=-1){
            m_scat_n_inputs = n_inputs;
            m_scat_J = J;
            m_scat_C = C;
            return *this;
        }

        /**
         * Use maxout for this layer.
         * @param n the number of maps to take maximum over (non-overlapping)
         */
        inline conv_layer_opts& maxout(int n){
            m_want_maxout = n > 1;
            m_maxout_N = n;
            return *this;
        }

        /**
         * Use dropout after pooling.
         * @param b if true, use dropout.
         */
        inline conv_layer_opts& dropout(bool b=true){
            m_want_dropout = b;
            return *this;
        }

        /**
         * Specify group name to be used for all ops inside the layer.
         * @param name the name for all conv-layer ops
         * @param unique if true, append a unique number to the name.
         */
        inline conv_layer_opts& group(std::string name="", bool unique=true) { m_group_name = name; return *this;  }

        /**
         * When initializing weights with this standard deviation.
         * @param std standard deviation
         */
        inline conv_layer_opts& weight_default_std(float std){ m_weight_default_std = std; return *this; }
    };


    /**
     * Implements a full-featured convolution layer.
     * 
     * All configuration options are specified in the conv_layer_opts class.
     */
    struct conv_layer
    : public model {
        public:
            typedef boost::shared_ptr<Op> op_ptr;
            typedef boost::shared_ptr<ParameterInput> input_ptr;
            typedef boost::shared_ptr<Sink> sink_ptr;

            input_ptr m_weights;
            input_ptr m_bias;
            op_ptr m_input;

            boost::shared_ptr<Noiser> m_noiser;

            op_ptr m_output, m_output_before_pooling, m_linear_output;
            bool m_verbose;

            float m_learnrate_factor, m_learnrate_factor_bias;
            float m_bias_default_value;
            float m_weight_default_std;

            int m_scat_n_inputs, m_scat_J, m_scat_C;

            conv_layer(const conv_layer_opts& cfg);

            /// empty ctor for serialization
            conv_layer():m_verbose(false){};

            inline conv_layer& set_learnrate_factor(float fW, float fB = -1.f){
            	m_learnrate_factor = fW;
                m_learnrate_factor_bias = (fB < 0) ? fW : fB;
            	return *this;
            }

            inline void set_noiser_active(bool b){
                if(m_noiser)
                    m_noiser->set_active(b);
            }
            void register_watches(monitor& mon);
            std::vector<Op*> get_params();
            void reset_params();

            virtual ~conv_layer();

        private:
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version){
                    ar & boost::serialization::base_object<model>(*this);
                    ar & m_weights & m_bias & m_output & m_verbose;
                    ar & m_scat_n_inputs & m_scat_J & m_scat_C;
                    ar & m_input;
                    if(version > 0){
                        ar & m_noiser;
                        ar & m_output_before_pooling;
                        ar & m_bias_default_value;
                    }
                    if(version > 2)
                        ar & m_linear_output;
                }
    };
} }

BOOST_CLASS_EXPORT_KEY(cuvnet::models::conv_layer)
BOOST_CLASS_VERSION(cuvnet::models::conv_layer, 3);

#endif /* __CUVNET_MODELS_ALEXCONV_LAYERS_HPP__ */
