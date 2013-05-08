#ifndef __CUVNET__MODEL_INIT_HPP__
#     define __CUVNET__MODEL_INIT_HPP__
#include <boost/shared_ptr.hpp>

namespace cuvnet
{
    class ParameterInput;
    namespace models
    {
        /**
         * initializes weights as in Glorot & Bengio, Dense Version.
         *
         * @param p the weight parameter (2D)
         * @param act_func_tanh if false, assume logistic activation function
         */
        void initialize_dense_glorot_bengio(boost::shared_ptr<ParameterInput> p, bool act_func_tanh=true);

        /**
         * initializes weights as in Glorot & Bengio, Version for Alex' convolution routines.
         *
         * @param p the weight parameter (2D)
         * @param act_func_tanh if false, assume logistic activation function
         */
        void initialize_alexconv_glorot_bengio(boost::shared_ptr<ParameterInput> p, unsigned int n_groups=1, bool act_func_tanh=true);

        /**
         * Set learnrate factors as advertised by LeCun
         *
         * Dense case:  "Because of possible correlations between input
         * variables, the learning rate of a unit should be inversely
         * proportional to the square root of the number of inputs to the
         * unit."
         *
         * @param W the weights for which learnrate is to be set
         * @param bias the bias for which learnrate is to be set
         * @param input_dim the axis of the weights whose shape reflects the dimensionality of the input
         */
        void set_learnrate_factors_lecun_dense(
                boost::shared_ptr<ParameterInput> W, unsigned int input_dim=0, 
                boost::shared_ptr<ParameterInput> bias = boost::shared_ptr<ParameterInput>());

        /**
         * Set learnrate factors as advertised by LeCun
         *
         * "If shared weights are used (as in TDNNs and
         * convolutional networks), the learning rate of
         * a weight should be inversely proportional to the
         * square root of the number of connection sharing
         * that weight."
         *
         * @param W the weights for which learnrate is to be set
         * @param b the bias for which learnrate is to be set
         * @param nshares the number of times one weight is shared.
         *                this should be proportional to the output
         *                map size.
         */
        void set_learnrate_factors_lecun_alexconv(
                boost::shared_ptr<ParameterInput> W, unsigned int nshares,
                boost::shared_ptr<ParameterInput> b = boost::shared_ptr<ParameterInput>());
    }
}

#endif /* __CUVNET__MODEL_INIT_HPP__ */
