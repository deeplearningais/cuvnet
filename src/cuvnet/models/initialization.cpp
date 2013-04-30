#include <cuv.hpp>
#include <cuvnet/ops/input.hpp>
#include "initialization.hpp"

namespace cuvnet
{
    namespace models
    {
        void initialize_dense_glorot_bengio(boost::shared_ptr<ParameterInput> p, bool act_func_tanh){
            float wnorm = p->data().shape(0)
                +         p->data().shape(1);
            float diff = std::sqrt(6.f/wnorm);
            if(!act_func_tanh)
                diff *= 4.f; 
            cuv::fill_rnd_uniform(p->data());
            p->data() *= diff*2.f;
            p->data() -= diff;
        }
    }
}
