#include <cuv.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/tools/logging.hpp>
#include "initialization.hpp"
#include <cmath>
#include <cerrno>
#include <fstream>

namespace {
	log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("initialization"));
}

namespace cuvnet
{
    namespace models
    {
        void initialize_dense_glorot_bengio(boost::shared_ptr<ParameterInput> p, bool act_func_tanh){
            float diff;
            if(act_func_tanh){
                float wnorm = p->data().shape(0)
                    +         p->data().shape(1);
                diff = std::sqrt(6.f/wnorm);
            }else{
                float fan_in = p->data().shape(0);
                diff = std::sqrt(3.f/fan_in);
            }
            cuv::fill_rnd_uniform(p->data());
            p->data() *= diff*2.f;
            p->data() -= diff;
        }

        void initialize_alexconv_scattering_transform(boost::format fmt, boost::shared_ptr<ParameterInput> _w,
        		unsigned int n_inputs, unsigned int J, unsigned int C, bool full, bool ones, bool neg){
        	cuvnet::matrix& w = _w->data();
            w = 0.0f;
            unsigned int n_filters = w.shape(2);
            unsigned int n_filt_pix = w.shape(1);

            std::string file = (fmt % (int)sqrt(n_filt_pix)).str();

            if(w.shape(0) < n_inputs){
            	LOG4CXX_WARN(g_log, "The supplied number of `useful' features ("<<n_inputs<<") in the inputs is larger than the real number of features ("<<w.shape(0)<<")in the input.");
            }

            unsigned int filter_bank_size = J * C;
            if(filter_bank_size > n_filters)
            {
            	LOG4CXX_WARN(g_log, "Loading only part of filter bank: "
            			"Would need: "<< filter_bank_size*n_inputs<<" outputs, but have only "<<n_filters);
            }

            if(full){
                // full processing: All inputs are filtered by the complete filter bank.
                boost::scoped_array<float> buf(new float[n_filt_pix]);
                unsigned int cnt = 0;
                for(unsigned int i = 0; i < n_inputs; i++){
                    std::ifstream ifs(file.c_str(), std::ios::binary);
                    for (unsigned int j = 0; j < J; j++){
                        for(unsigned int c = 0; c < C; c++){
                            if(++cnt > n_filters)
                                break;
                            for (unsigned int cmplx = 0; cmplx < 2; ++cmplx) {
                                bool res = ifs.read((char*)buf.get(), n_filt_pix * sizeof(float));
                                if(!res){
                                    std::stringstream ss;
                                    ss << "I/O error while loading filters: " << std::strerror(errno);
                                    throw std::runtime_error(ss.str());
                                }
                                for (unsigned int k = 0; k < n_filt_pix; ++k) {
                                    if(!ones)
                                        w(i, k, (i * J * C * 2) + (j * C * 2) + (c * 2) + cmplx) = buf[k];
                                    else
                                        w(i, k, (i * J * C * 2) + (j * C * 2) + (c * 2) + cmplx) = 1.f;
                                }
                            }
                        }
                    }
                }
            }else{
                // "Interleaved" processing: Each output scale filters only one
                // input scale, namely, the one with the same index.
                boost::scoped_array<float> buf(new float[filter_bank_size * 2 * n_filt_pix]);
                {
                    // read complete filter bank
                    std::ifstream ifs(file.c_str(), std::ios::binary);
                    bool res = ifs.read((char*)buf.get(), sizeof(float) * (filter_bank_size * 2 * n_filt_pix));
                    if(!res){
                        std::stringstream ss;
                        ss << "I/O error while loading filters: " << std::strerror(errno);
                        throw std::runtime_error(ss.str());
                    }
                }
                unsigned int cnt = 0;
                for(unsigned int i = 0; i < n_inputs; i++){
                    assert(J == 2);
                    unsigned int i_j = i / (n_inputs/2); // first half has j=0
                    unsigned int i_c = i % C;
                    cuvAssert(i_j < J);
                    for(unsigned int c = 0; c < C; c++){
                        if(2 * cnt + 2 > n_filters)
                            // don't try to generate more filters than existing in the weight matrix
                            break;
                        for (unsigned int cmplx = 0; cmplx < 2; ++cmplx) {
                            for (unsigned int k = 0; k < n_filt_pix; ++k) {
                                float val = buf[i_j * C * 2 * n_filt_pix + c * 2 * n_filt_pix + cmplx * n_filt_pix + k];
                                if(ones)
                                    w(i, k, 2*cnt + cmplx) = 1.f;
                                else
                                    w(i, k, 2*cnt + cmplx) = val;
                            }

                            if(neg){
                                // add a second input to the current input,
                                // which is rotated by 90 degrees
                                unsigned int c2 = (i_c + C/2) % C;
                                unsigned int i2 = C * (i / C);  // same J; same group of Cs
                                i2 += c2; // go to orthogonal orientation
                                for (unsigned int k = 0; k < n_filt_pix; ++k) {
                                    if(ones)
                                        w(i2, k, 2*cnt + cmplx) =  1.f;
                                    else
                                        w(i2, k, 2*cnt + cmplx) =  -buf[i_j * C * 2 * n_filt_pix + c * 2 * n_filt_pix + cmplx * n_filt_pix + k];
                                }
                            }
                        }
                        cnt ++;
                    }
                }
            }
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
            W->set_learnrate_factor(1.f / std::sqrt((float)n_inputs));
            b->set_learnrate_factor(1.f);
            b->set_weight_decay_factor(0.f);
        }

        void set_learnrate_factors_lecun_conv(
                boost::shared_ptr<ParameterInput> W, unsigned int nshares,
                boost::shared_ptr<ParameterInput> b = boost::shared_ptr<ParameterInput>()){
            W->set_learnrate_factor(1.f / std::sqrt((float)nshares));
            b->set_learnrate_factor(1.f / std::sqrt((float)nshares));
            b->set_weight_decay_factor(0.f);
        }
    }
}
