#ifndef __OP__WEIGHTED_SUBTENSOR_HPP__
#     define __OP__WEIGHTED_SUBTENSOR_HPP__

#include <cmath>
#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <log4cxx/logger.h>
namespace cuvnet
{
    class monitor; // used in WeightedSubtensor
        /**
         * Calculates op out of consecutive elements in the input.
         * The input must be of shape n_maps x img_data x n_batch or of shape n_maps x img_data_X x img_data_Y x n_batch
         *
         * Expressed in numpy style, this calculates:
         *
         * f(X) = op((W[::2]* X[::2, ...], W[1::2]* X[1::2, ...], .. )
         *
         * @ingroup Ops
         */
        class WeightedSubtensor
            : public Op{
                public:
                    typedef Op::value_type    value_type;
                    typedef Op::op_ptr        op_ptr;
                    typedef Op::value_ptr     value_ptr;
                    typedef Op::param_t       param_t;
                    typedef Op::result_t      result_t;

                    // for profiling host / dev
                    typedef cuv::tensor<unsigned char,cuv::dev_memory_space> char_matrix;
                    unsigned int m_size;
                    unsigned int m_subspace_size;
                    unsigned int m_stride;
                    cow_ptr<char_matrix>  m_max_idx;
                    value_ptr    m_lae;
                    boost::shared_ptr<cuvnet::monitor> m_S;     
                    cuv::alex_conv::weighted_subtensor_functor m_to;
                    float m_eps;
                    bool m_spn;
                    bool m_memory_flag;
                private:
                public:
                    WeightedSubtensor() :Op(2,1){} ///< for serialization
                    /**
                     * ctor.
                     * @param images the input images
                     */
                    WeightedSubtensor(result_t& images, result_t& m_W, unsigned int size, unsigned int stride, 
                            unsigned int subspace_size, cuv::alex_conv::weighted_subtensor_functor to, float eps, bool spn)
                        :Op(2,1),
                        m_size(size),
                        m_subspace_size(subspace_size),
                        m_stride(stride),
                        m_to(to),
                        m_eps(eps),
                        m_spn(spn)
                    {
                        add_param(0,images);
                        add_param(1,m_W);

                        using namespace cuv;
                           //generate dummy tensor ( empty ) to avoid null pointer exceptions
                        value_ptr z(new value_type(0, value_ptr::s_allocator));
                        m_lae = z;
                        
                        //m_S = S;
                        m_memory_flag = false;
                        //generate dummy tensor ( empty ) to avoid null pointer exceptions
                        cow_ptr<char_matrix> m(new char_matrix(0, value_ptr::s_allocator));
                        m_max_idx = m;                          
                    }

                    void fprop();
                    void bprop();

                    void _determine_shapes();
                    virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                    void set_S(boost::shared_ptr<cuvnet::monitor> S){
                        m_S = S;
                    }
                                       
                private:
                    friend class boost::serialization::access;
                    template<class Archive>
                        void serialize(Archive& ar, const unsigned int version){
                            ar & boost::serialization::base_object<Op>(*this) & m_subspace_size & m_size & m_to & m_stride & m_eps & m_spn;
                        }
            };
}
#endif /* __OP__WEIGHTED_SUBTENSOR_HPP__ */
