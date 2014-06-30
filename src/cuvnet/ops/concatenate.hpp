#ifndef __OP_CONCATENATE_HPP__
#     define __OP_CONCATENATE_HPP__

#include <cuvnet/op.hpp>
#include <boost/limits.hpp>
namespace cuvnet
{       
     /**
     * Concatenates N input tensors
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
                std::vector<std::vector<int> > m_pi_shape;
                unsigned int m_dim;
                unsigned int m_n;
                bool m_reshape;
                std::vector<unsigned int> m_tmp_shape;
            public:
                Concatenate() :Op(2,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param dim the dimension along which the inputs are concatenated
                 */
                Concatenate(boost::shared_ptr<std::vector<result_t> > & in, unsigned int dim)
                    :Op(in->size(),1),
                    m_dim(dim),
                    m_n(in->size())
                {
                    //add all n params
                    for ( unsigned int i = 0; i < in->size() ; i++) add_param(i, (*in)[i]);
                }
                void fprop();
                void bprop();

                void _determine_shapes();
                value_type get_subtensor(const value_type &v, unsigned int position);
            protected:
                    boost::shared_ptr<std::vector<unsigned int> > get_pi_shape(value_type & vi);
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_dim;
                        ar & m_n;
                    }
        };
}

#endif /* __OP_CONCATENATE_HPP__ */
