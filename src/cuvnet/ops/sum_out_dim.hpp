#ifndef __OP_SUM_OUT_DIM_HPP__
#     define __OP_SUM_OUT_DIM_HPP__

#include <cuvnet/op.hpp>
#include <numeric>
namespace cuvnet
{
    /**
     * Sums out a dimension (currently only first or last dimension can be summed out.
     *
     * E.g. a (3,4,5)-sized input summed to dimension 0 results in a
     * 3-dimensional vector with size (1,4,5).
     *
     * @ingroup Ops
     *
     */
    class Sum_Out_Dim
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                unsigned int m_axis;
                unsigned int m_ndim;
                std::vector<unsigned int> m_param_reshape;
                std::vector<unsigned int> m_param_shape;
                std::vector<unsigned int> m_res_reshape;
                std::vector<unsigned int> m_res_shape;
                float m_n_summed;
                bool m_mean;
                bool m_squared;
            public:
                Sum_Out_Dim() :   Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param mat the n-dimensional array
                 * @param axis currently, either 0 or n-1 is allowed
                 * @param mean if true, divide by the number of entries summed over
                 * @param squared if true, sum squared elements
                 */
                Sum_Out_Dim(result_t& mat, unsigned int axis, bool mean=false, bool squared=false)
                    :   Op(1,1)
                      , m_axis(axis)
                      , m_mean(mean)
                      , m_squared(squared)
            {
                add_param(0,mat);
            }

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                void fprop();
                void bprop();
                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                        ar & m_mean;
                        ar & m_squared;
                        ar & m_param_reshape;
                        ar & m_param_shape;
                        ar & m_res_reshape;
                        ar & m_res_shape;
                        
                    }
    };

}
#endif /* __OP_SUM_OUT_DIM_HPP__ */
