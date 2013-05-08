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

        void initialize_alexconv_glorot_bengio(boost::shared_ptr<ParameterInput> p, unsigned int n_groups, bool act_func_tanh){
            // we need to calculate fan-in and fan-out to a convolutional layer. 
            // However, we do not have all information at this position,
            // therefore, we make some assumptions about fan-out.
            unsigned int n_srcmaps = p->data().shape(0);
            unsigned int n_fltpix = p->data().shape(1);
            unsigned int fan_in = n_srcmaps / n_groups * n_fltpix;

            // the primary things we don't know are the number of maps in the
            // layer above the current destination layer and the filter size
            // used to get there. Lets assume the /ratio/ of the numer of maps
            // stays the same, and the filter size is constant.
            unsigned int n_dstmaps = p->data().shape(2);
            unsigned int n_dst2maps = n_dstmaps * n_dstmaps / n_srcmaps;
            unsigned int fan_out = std::min(2u, n_dst2maps / n_groups * n_fltpix);

            float wnorm = fan_in + fan_out;
            float diff = std::sqrt(6.f/wnorm);
            if(!act_func_tanh)
                diff *= 4.f; 
            cuv::fill_rnd_uniform(p->data());
            p->data() *= diff*2.f;
            p->data() -= diff;
        }

        void set_learnrate_factors_lecun_dense(
                boost::shared_ptr<ParameterInput> W, unsigned int input_dim,
                boost::shared_ptr<ParameterInput> b){
            cuvAssert(W->data().ndim()==2);
            cuvAssert(b->data().ndim()==1);
            unsigned int n_inputs = W->data().shape(input_dim);
            W->set_learnrate_factor(1.f / sqrt((float)n_inputs));
            b->set_learnrate_factor(1.f);
            b->set_weight_decay_factor(0.f);
        }

        void set_learnrate_factors_lecun_conv(
                boost::shared_ptr<ParameterInput> W, unsigned int nshares,
                boost::shared_ptr<ParameterInput> b = boost::shared_ptr<ParameterInput>()){
            W->set_learnrate_factor(1.f / sqrt((float)nshares));
            b->set_learnrate_factor(1.f / sqrt((float)nshares));
            b->set_weight_decay_factor(0.f);
        }
    }
}
