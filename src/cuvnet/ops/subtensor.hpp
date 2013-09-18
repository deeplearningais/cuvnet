#ifndef __OP_SUBTENSOR_HPP__
#     define __OP_SUBTENSOR_HPP__

#include <cuvnet/op.hpp>
#include <boost/limits.hpp>
namespace cuvnet
{

    /**
     * Gets the subtensor from the input tensor.
     *
     * 
     * @ingroup Ops
     */
    class Subtensor 
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                std::vector<int> m_starts;
                std::vector<int> m_ends;
                std::vector<int> m_starts_det;
                std::vector<int> m_ends_det;
                std::vector<bool> m_is_degen;
                bool m_copy;
                
            public:
                Subtensor() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                template<int D, int E>
                Subtensor(result_t& in, const cuv::index_gen<D,E>& idx, bool copy = true)
                    :Op(1,1),
                    m_starts(D),
                    m_ends(D),
                    m_starts_det(D),
                    m_ends_det(D),
                    m_is_degen(D),
                    m_copy(copy)
                {
                    add_param(0,in);
                    for(unsigned int i=0; i<D; i++){
                        m_starts[i]  = idx.ranges_[i].get_start(0);
                        m_ends[i] = idx.ranges_[i].get_finish(std::numeric_limits<int>::min());
                        if(idx.ranges_[i].is_degenerate()){
                            m_is_degen[i] = true;
                        }
                        else{
                            m_is_degen[i] = false;
                        }

                    }
                }
                void fprop();
                void bprop();

                void _determine_shapes();
                void get_subtensor(value_type &v);
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_starts & m_ends & m_starts_det & m_ends_det & m_is_degen & m_copy;
                    }
        };
}

#endif /* __OP_SUBTENSOR_HPP__ */
