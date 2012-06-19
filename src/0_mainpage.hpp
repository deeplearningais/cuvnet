/**
 *
 * @defgroup OpGroup Symbolic operations
 * @defgroup models Models, which demonstrate how \c Ops can be combined.
 * @defgroup tools Everything not connected to the symbolic operations themselves
 *
 * @addtogroup OpGroup
 * @{
 *   @defgroup Ops Implementations of specific symbolic operations
 *   @defgroup op_visitors Operations on symbolic functions
 * @}
 * @addtogroup tools
 * @{
 *   @defgroup preproc Pre-processing of datasets
 *   @defgroup learning Parameter learning
 * @}
 *
 * @defgroup datasets Dataset accessors
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
