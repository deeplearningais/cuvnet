/**
 *
 * @defgroup OpGroup Symbolic operations
 * @defgroup models Models, which demonstrate how Ops can be combined.
 * @defgroup tools Everything not connected to the symbolic operations themselves
 *
 * @addtogroup OpGroup
 * @{
 *   @defgroup Ops Implementations of specific symbolic operations
 *   @addtogroup Ops
 *   @{
 *      @defgroup CaffeOps Operators related to the caffe library
 *      @defgroup TheanoOps Operators related to the Theano library
 *      @defgroup CudaConvnetOps Operators or related to Alex Krizhevskys Cuda-Convnet library
 *      @defgroup CuDNNOps Operators related to NVIDIAs cuDNN library
 *   @}
 *   @defgroup convenience_funcs Convenience functions for creating complex symbolic operations
 *   @defgroup op_visitors Operations on symbolic functions
 *   @defgroup serialization Helpers for serialization of symbolic functions and models
 * @}
 * @addtogroup tools
 * @{
 *   @defgroup preproc Pre-processing of datasets
 *   @defgroup learning Parameter learning
 *   @addtogroup learning
 *   @{
 *     @defgroup gd Gradient Descent
 *     @defgroup gd_util Gradient Descent Utilities
 *     @defgroup learning_exceptions Learning Exceptions
 *   @}
 *   @defgroup python_helpers Helper functions for accessing cuvnet functionality from Python
 *   @defgroup netcom Helpers for performing asynchronous stochastic gradient descent
 * @}
 *
 * @defgroup datasets Dataset accessors
 * @addtogroup datasets
 * @{
 *   @defgroup bbtools Tools for dealing with bounding boxes
 * @}
 *
 * @mainpage
 *
 * @section summary  Summary
 *
 * Cuvnet is a library of symbolic operations.
 * 
 * Symbolic operations operate on matrices and can be executed by on CPU or on
 * GPU. For this purpose, \c cuvnet makes heavy use of the \c cuv library.
 *
 * Functions defined by symbolic operations cannot only be evaluated
 * efficiently, the derivative can also be calculated without specified by the
 * user. This is similar to \c theano, which is developed at university
 * Montreal, and \c eblearn (university New York). However, \c cuvnet does not
 * use code generation, so it is much easier to debug.
 *
 * @section features  Features
 *
 * - Works on CPU and GPU
 * - Small code base, since it relies on tested CUV library for all the `dirty'
 *   stuff
 * - Automatic differentiation
 * - Space-efficient function evaluation and backprop (reuses data where
 *   possible and avoids copies using copy-on-write)
 * - Tools for learning (datasets, preprocessing, gradient descent,
 *   cross-validation, ...)
 * - Fast convolutions curtesy of Alex Krizhevsky (much faster than Theano's)
 *
 * @section motivation Motivation
 *
 * A motivating example is e.g. the construction of a convolutional LeNet-type
 * classifier network from almost scratch (taken from \c cuvnet/models/lenet.hpp):
 @code
    hl1 =
        local_pool(
                tanh(
                    mat_plus_vec(
                        convolve( 
                            reorder_for_conv(inputs),
                            m_conv1_weights, false),
                        m_bias1, 0)),
                cuv::alex_conv::PT_MAX); 

    hl2 = 
        local_pool(
                tanh(
                    mat_plus_vec(
                        convolve( 
                            hl1,
                            m_conv2_weights, false),
                        m_bias2, 0)),
                cuv::alex_conv::PT_MAX); 
    hl2 = reorder_from_conv(hl2);
    hl2 = reshape(hl2, cuv::extents[batchsize][-1]);
    hl3 =
        tanh(
        mat_plus_vec(
            prod(hl2, m_weights3),
            m_bias3,1));

    loss = boost::make_shared<regression_type>(hl3, target)->loss();

@endcode
 *
 * @section deps Dependencies
 *
 * You have to install the following to get cuvnet to work:
 * - cuv https://github.com/deeplearningais/CUV
 * - mdbq https://github.com/temporaer/MDBQ
 * - libjpeg-dev
 * - liblog4cxx10-dev
 * - python-dev
 *
 *
 * @section compiling Compiling
 *
 * download cuvnet,
 * @code
 * $ cd cuvnet
 * $ mkdir -p build/release
 * $ cd build/release
 * $ cmake -DCMAKE_BUILD_TYPE=Release ../../
 * $ make -j
 * $ ./src/test/cuvnet_test  # run tests
 * @endcode
 *
 *
 * @section contact  Contact
 *
 * We are eager to help you getting started with cuvnet and improve the library continuously!
 * If you have any questions, feel free to contact Hannes Schulz (schulz at ais dot uni-bonn dot de).
 * You can find the website of our group at http://www.ais.uni-bonn.de/deep_learning/index.html.
 *
 */


/** 
 * @namespace cuvnet
 * @brief contains most cuvnet functionality
 */

namespace cuvnet{
}

/**
 * @example linear_regression.cpp very simple demonstration of how to use cuvnet
 * @example logistic_regression.cpp slightly more advanced concepts: sinks
 * @example minibatch_learning.cpp slightly more advanced concepts: callbacks for online-learning
 * @example hyperopt.cpp how to use James Bergstra's hyperopt for hyper parameter optimization
 */
