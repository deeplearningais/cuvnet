#ifndef __OP_THEANO_OPS_HPP__
#     define __OP_THEANO_OPS_HPP__

#include <cuv/libs/theano_ops/theano_ops.hpp>
#include <cuvnet/op.hpp>
namespace cuvnet
{
    /**
     *  Theano operator, fliping the dimensions 
     *
     * @ingroup Ops
     * 
     */
    class FlipDims
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                std::vector<bool> m_pattern;

            public:
                FlipDims() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                template<std::size_t D>
                FlipDims(result_t& in, const cuv::extent_gen<D>& eg)
                    :Op(1,1),
                    m_pattern(D)
                {
                    add_param(0,in);
                    for(unsigned int i=0; i<D; i++){
                        m_pattern[i] = eg.ranges_[i].finish();
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
                        ar & m_pattern;
                    }
        };

    /**
     *  Theano operator, shuffling the dimensions 
     *
     * @ingroup Ops
     * 
     */
    class ShuffleDim 
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                std::vector<int> m_pattern;
                unsigned int m_ndim;

            public:
                ShuffleDim() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                template<std::size_t D>
                ShuffleDim(result_t& in, const cuv::extent_gen<D>& eg)
                    :Op(1,1),
                    m_pattern(D)
                {
                    add_param(0,in);
                    for(unsigned int i=0; i<D; i++){
                        m_pattern[i] = eg.ranges_[i].finish();
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
                        ar & m_pattern;
                    }
        };

}

#endif /* __OP_THEANO_OPS_HPP__ */
