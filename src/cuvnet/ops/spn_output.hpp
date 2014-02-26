#ifndef __OP_SPN_OUTPUT_HPP__
#     define __OP_SPN_OUTPUT_HPP__

#include <cmath>
#include <cuvnet/op.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>

#include <log4cxx/logger.h>

namespace cuvnet
{
        /**
         * calculates the output layer of an spn
         * input X of shape classes x regions x batch
         * it calculates sum_i=1^c exp(X[i, :, b ] + Y[b][i] + W[i]) 
         *
         * dst will be of shape b x regions
         * @ingroup Ops
         */
        class Spn_Output_Op
            : public Op{
                public:
                    typedef Op::value_type    value_type;
                    typedef Op::op_ptr        op_ptr;
                    typedef Op::value_ptr     value_ptr;
                    typedef Op::param_t       param_t;
                    typedef Op::result_t      result_t;
                    typedef cuv::tensor<unsigned int,cuv::dev_memory_space> int_mat;                    

                    boost::shared_ptr<cuvnet::monitor> m_S;     
                    unsigned int m_classes;
                    bool m_hard_gd;
                    float m_eps;
                    value_ptr    m_lae;
                    cow_ptr<int_mat> m_max_idx;
                    bool m_memory_flag;
                private:
                    inline Op::value_ptr get_data_ptr(bool can_overwritem, param_t::element_type* p);
                public:
                    Spn_Output_Op() :Op(3,1){} ///< for serialization
                    /**
                     * ctor.
                     * @param images the input images
                     */
                    Spn_Output_Op(result_t& images, result_t& m_W, result_t& Y, boost::shared_ptr<cuvnet::monitor> S, unsigned int classes, bool hard_gd, float eps)
                        :Op(3,1),
                        m_classes(classes),
                        m_hard_gd(hard_gd),                       
                        m_eps(eps)
                    {
                        add_param(0,images);
                        add_param(1,m_W);
                        add_param(2,Y);
                        m_S = S;      
                        
                        //generate dummy tensor ( empty ) to avoid null pointer exceptions
                        value_ptr z(new value_type(cuv::extents[1], value_ptr::s_allocator));
                        m_lae = z;
                        
                        if (hard_gd){
                            cow_ptr<int_mat> m(new int_mat(cuv::extents[1], value_ptr::s_allocator));
                            m_max_idx = m; 
                            m_memory_flag = false;
                        }
                    }

                    void fprop();
                    void bprop();
                    virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                    void _determine_shapes();

                    void set_S(boost::shared_ptr<cuvnet::monitor> S){
                        m_S = S;
                    }
                                       
                private:
                    friend class boost::serialization::access;
                    template<class Archive>
                        void serialize(Archive& ar, const unsigned int version){
                            ar & boost::serialization::base_object<Op>(*this) & m_classes & m_eps;
                        }
            };

}
#endif /* __OP_SPN_OUTPUT_HPP__ */
