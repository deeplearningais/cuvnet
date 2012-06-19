#ifndef __OP_IDENTITY_HPP__
#     define __OP_IDENTITY_HPP__

namespace cuvnet
{
    /**
     * simply passes on its inputs to whoever wants it.
     *
     * This is mainly an example of how to write an \c Op, it is not really
     * useful.
     *
     * @ingroup Ops
     */
    class Identity 
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Identity(){} /// for serialization
                Identity(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop(){
                    // identity
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    if(r0.can_add_directly()){
                        value_ptr& ptr = r0.overwrite_or_add_value();
                        *ptr          += p0.value.cdata();
                    }else{
                        r0.push(p0.value); // 'copy' a newly created matrix
                    }
                    p0.value.reset();       // don't need that for backprop etc.
                }
                void bprop(){
                    // identity
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    if(p0.can_add_directly()){
                        value_ptr& ptr = p0.overwrite_or_add_value();
                        *ptr          += r0.delta.cdata();
                    }else{
                        r0.push(p0.value); // 'copy' a newly created matrix
                    }
                    r0.delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}
#endif /* __OP_IDENTITY_HPP__ */
