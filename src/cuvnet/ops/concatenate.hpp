#ifndef __OP_CONCATENATE_HPP__
#     define __OP_CONCATENATE_HPP__

#include <cuvnet/op.hpp>
#include <boost/limits.hpp>
namespace cuvnet
{

    /**
     * Concatenates two input tensors
     *      
     * 
     * @ingroup Ops
     */
    class Concatenate
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                std::vector<int> m_p0_shape;
                std::vector<int> m_p1_shape;
                unsigned int m_dim;
            public:
                Concatenate() :Op(2,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                Concatenate(result_t& in_1, result_t& in_2, unsigned int dim)
                    :Op(2,1),
                    m_dim(dim)
                {
                    assert(dim <=2);
                    add_param(0,in_1);
                    add_param(1,in_2);
                }
                void fprop();
                void bprop();

                void _determine_shapes();
                value_type get_subtensor(const value_type &v, bool first);
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
        
            /**
     * Concatenates N input tensors
     *      
     * 
     * @ingroup Ops
     */
    class Concatenate_N
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                std::vector<std::vector<int> > m_pi_shape;
                unsigned int m_dim;
                unsigned int m_n;
            public:
                Concatenate_N() :Op(2,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                Concatenate_N(std::vector<op_ptr> & in, unsigned int dim, unsigned int n)
                    :Op(n,1),
                    m_dim(dim),
                    m_n(n)
                {
                    assert(dim <=2);
                    //add all n params
                    for ( unsigned int i = 0; i < n ; i++) add_param(i, in[i]->result());
                }
                void fprop();
                void bprop();

                void _determine_shapes();
                value_type get_subtensor(const value_type &v, unsigned int position);
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}

#endif /* __OP_CONCATENATE_HPP__ */
