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

                    boost::shared_ptr<cuvnet::monitor> m_S;     
                    unsigned int m_classes;
                    float m_eps;
                private:
                    inline Op::value_ptr get_data_ptr(bool can_overwritem, param_t::element_type* p);
                public:
                    Spn_Output_Op() :Op(3,1){} ///< for serialization
                    /**
                     * ctor.
                     * @param images the input images
                     */
                    Spn_Output_Op(result_t& images, result_t& m_W, result_t& Y, boost::shared_ptr<cuvnet::monitor> S, unsigned int classes, float eps)
                        :Op(3,1),
                        m_classes(classes),
                        m_eps(eps)
                    {
                        add_param(0,images);
                        add_param(1,m_W);
                        add_param(2,Y);
                        m_S = S;                      
                    }

                    void fprop();
                    void bprop();

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
