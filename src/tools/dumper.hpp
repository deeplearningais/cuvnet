#ifndef __DUMPER_HPP__
#     define __DUMPER_HPP__

#include<boost/signals.hpp>
#include<boost/bind.hpp>
#include<boost/limits.hpp>
#include<cuvnet/op.hpp>
#include<cuvnet/op_utils.hpp>
#include<cuvnet/ops/output.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>
#include<cuv/tensor_ops/rprop.hpp>

namespace cuvnet
{

    /**
     * processes data through a function and writes the outputs to a stream.
     *
     * @deprecated
     * @ingroup tools
     */
    struct dumper{
        public:
            typedef std::vector<Op*> paramvec_t;
        protected:
            swiper           m_swipe;    ///< does fprop and bprop for us
            boost::shared_ptr<Sink> m_sink; ///< data is read from here
        public:
            /// triggered before executing a batch (you should load batch data here!)
            boost::signal<void(unsigned int,unsigned int)> before_batch;

            /// should return current number of batches
            boost::signal<unsigned int(void)> current_batch_num;

            /**
             * constructor
             * 
             */
            dumper(Op::op_ptr op, unsigned int result=0)
                :m_swipe(*op, result, paramvec_t())
                ,m_sink(new Sink("dumper", op->result(result)))
            { }

            /**
             * (virtual) destructor
             */
            virtual ~dumper(){
                m_sink->detach_from_params();
                m_sink->detach_from_results();
                m_sink.reset();
            }

            /**
             * Dumps the result of a function call to an output stream
             */
            void dump(std::ostream& o){
                unsigned int n_batches = current_batch_num();
                for (unsigned int  batch = 0; batch < n_batches; ++batch) {
                    before_batch(0, batch);
                    m_swipe.fprop();
                    // copy data to host
                    cuv::tensor<float,cuv::host_memory_space> tmp = m_sink->cdata(); 
                    // dump to ostream
                    o.write((char*)tmp.ptr(), sizeof(float)*tmp.size());
                }
            }
    };
}

#endif /* __DUMPER_HPP__ */
