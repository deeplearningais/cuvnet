#ifndef __CUVNET__MODEL_INIT_HPP__
#     define __CUVNET__MODEL_INIT_HPP__
#include <boost/shared_ptr.hpp>

namespace cuvnet
{
    class ParameterInput;
    namespace models
    {
        /**
         * initializes weights as in Glorot & Bengio.
         *
         * @param p the weight parameter (2D)
         * @param act_func_tanh if false, assume logistic activation function
         */
        void initialize_dense_glorot_bengio(boost::shared_ptr<ParameterInput> p, bool act_func_tanh=true);
    }
}

#endif /* __CUVNET__MODEL_INIT_HPP__ */
