#ifndef __CUVNET__MODEL_INIT_HPP__
#     define __CUVNET__MODEL_INIT_HPP__
#include <boost/format.hpp>
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
         * load a filter bank into a weight matrix.
         *
         * @param file the filename to load filter bank from, with a %d in place of the filter size
         * @param w the weight matrix to load weights to
         * @param J the number of scales in the filter bank
         * @param C the number of angles in the filter bank
         * @param full if true, process all inputs with complete filter bank.
         *             Otherwise, process each input with the corresponding scale in the filterbank.
         *             Assuming the image has been downsampled in between, this
         *             corresponds to the "next octave".
         * @param ones if true, only put a 1 in positions where the filterbank
         *             would end up, creating a mask that can be used to censor
         *             weight updates
         * @param neg if true, use combination of positive and negative filters to create
         *             (high-level texture) edge filters
         */
        void initialize_alexconv_scattering_transform(boost::format fmt, boost::shared_ptr<ParameterInput> _w, unsigned int n_inputs, unsigned int J, unsigned int C, bool full, bool ones, bool neg);


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
