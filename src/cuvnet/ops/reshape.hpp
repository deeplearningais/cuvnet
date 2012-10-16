#ifndef __OP_RESHAPE_HPP__
#     define __OP_RESHAPE_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * Similar to Reshape(), but the shape is inferred from the shape of the
     * parameter.
     * 
     * \b Parameters:
     * - First parameter is the value that needs reshaping,
     * - Second (optional) parameter is the number of dimensions in the returned variable.
     *
     * \b Returns a variable with the same shape as x in the leading outdim-1
     * dimensions, but with all remaining dimensions of x collapsed into the
     * last dimension.
     *
     * For \b example, if we flatten a tensor of shape (2,3,4,5) with flatten(x,
     * outdim=2), then we’ll have the same (2-1=1) leading dimensions (2,), and
     * the remaining dimensions are collapsed. So the output in this example
     * would have shape (2, 60).
     *
     * @ingroup Ops
     * 
     */
    class Flatten
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_outdim;
            public:
                Flatten() :Op(1,1){} // for serialization
                Flatten(result_t& in, unsigned int outdim=1)
                    :Op(1,1),
                    m_outdim(outdim)
                {
                    add_param(0,in);
                }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_outdim;
                    }
        };

    /**
     * Ensures that output has certain shape.
     *
     * one component can be <0, which means that the shape there is deduced
     * from the input dimensions.
     * 
     * @ingroup Ops
     */
    class Reshape
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                std::vector<int> m_shape;
            public:
                Reshape() :Op(1,1){} // for serialization
                template<std::size_t D>
                Reshape(result_t& in, const cuv::extent_gen<D>& eg)
                    :Op(1,1),
                    m_shape(D)
                {
                    add_param(0,in);
                    for(unsigned int i=0; i<D; i++){
                        m_shape[i] = eg.ranges_[i].finish();
                    }
                }
                void fprop();
                void bprop();

                void _determine_shapes();

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_shape;
                    }
        };
}

#endif /* __OP_RESHAPE_HPP__ */
