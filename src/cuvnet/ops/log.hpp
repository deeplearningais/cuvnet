#ifndef __OP_LOG_HPP__
#     define __OP_LOG_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    class Log
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Log(){} /// for serialization
                Log(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();
                    value_ptr res(new value_type(inp.shape()));

                    apply_scalar_functor( *res,
                            inp, SF_LOG);

                    r0.push(res); // 'copy' a newly created matrix

                    if(!p0.need_derivative)
                        p0.value.reset();       // forget it
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    // try overwriting p0
                    const value_type& inp = p0.value.cdata();
                    value_type&       out = p0.value.data_onlyshape();
                    apply_scalar_functor(out,inp,SF_INV);
                    out *= r0.delta.cdata();
                    r0.delta.reset();
                    p0.push(p0.value);
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}
#endif /* __OP_LOG_HPP__ */
