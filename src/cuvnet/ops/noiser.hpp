#ifndef __OP_NOISER_HPP__
#     define __OP_NOISER_HPP__
#include <cuvnet/op.hpp>

namespace cuvnet
{
    class Noiser
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                float m_std;

            public:
                Noiser(){} /// for serialization
                Noiser(result_t& p0, float std=1.f)
                    :Op(1,1), m_std(std)
                     {
                         add_param(0,p0);
                     }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    //const value_type& inp0 = p0.value.cdata();           // original

                    if(r0.can_overwrite_directly()){
                        value_ptr& v  = r0.overwrite_or_add_value();
                        *v            = p0.value.cdata(); 
                        cuv::add_rnd_normal(*v,m_std);
                    }
                    else if(r0.can_add_directly()){
                        value_ptr& v = r0.overwrite_or_add_value();
                        *v += p0.value.cdata();
                        cuv::add_rnd_normal(*v,m_std);
                        p0.value.reset(); // forget it
                    }
                    else{
                        value_ptr v = p0.value; // copy p0
                        p0.value.reset();       // try to overwrite r0
                        cuv::add_rnd_normal(*v,m_std);
                        r0.push(v);
                    }

                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    value_ptr delta_orig = r0.delta;
                    if(p0.can_add_directly()){
                        p0.overwrite_or_add_value().data() += r0.delta.cdata();
                    }else if(p0.can_overwrite_directly()){
                        if(p0.overwrite_or_add_value().unique())
                            p0.overwrite_or_add_value() = r0.delta;
                        else
                            p0.overwrite_or_add_value().data_onlyshape() = r0.delta.cdata();
                    }else{
                        p0.push(r0.delta);
                    }
                    r0.delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_std;
                    }
        };
}
#endif /* __OP_NOISER_HPP__ */
