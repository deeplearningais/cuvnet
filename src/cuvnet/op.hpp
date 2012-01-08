// vim:ts=4:sw=4:et:
#ifndef __OP_HPP__
#     define __OP_HPP__
#include <list>
#include <map>
#include <boost/weak_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/base_object.hpp>
#include <cuvnet/common.hpp>
#include <cuvnet/smart_ptr.hpp>
#include <cuvnet/detail/op_result.hpp>
#include <cuvnet/detail/op_param.hpp>

namespace cuvnet
{
    class Op
        : public boost::enable_shared_from_this<Op>{
            public:
                typedef matrix value_type;
                typedef cow_ptr<value_type>                 value_ptr;
                typedef boost::shared_ptr<Op>               op_ptr;
                typedef boost::weak_ptr<detail::op_param<value_type> >     weak_param_t;
                typedef boost::shared_ptr<detail::op_param<value_type> >        param_t;
                typedef boost::shared_ptr<detail::op_result<value_type> >      result_t;
            protected:
                std::vector<param_t>  m_params;
                std::vector<result_t> m_results;

            public:
                /**
                 * Default constructor (you shouldn't use this, only implicitly during deserialization!)
                 */
                Op();
                Op(unsigned int n_params, unsigned int n_results);
                ~Op();

                Op& detach_from_params();
                Op& detach_from_results();

                /** 
                 * returns a reference to the i-th result
                 *
                 * @note the included check is inefficient but avoids making constructor private
                 *       since we cannot determine `shared_from_this' in
                 *       the constructor when we construct objects in m_results.
                 *       I assume that result() will be primarily used to construct
                 *       functions, which is not that often.
                 */
                result_t&       result(const unsigned int i=0);
                /** 
                 * returns a reference to the i-th parameter
                 */
                param_t&       param(const unsigned int i=0);
                void set_n_params(unsigned int n);
                void set_n_results(unsigned int n);
                unsigned int get_n_params(){ return m_params.size(); }
                unsigned int get_n_results(){ return m_results.size(); }
                void add_param(unsigned int idx, result_t& p);

                /**
                 * calculate recursively what needs to be calculated to
                 * derive this operator w.r.t. a set of parameters.
                 *
                 * The results are stored in the function itself.
                 *
                 * @param l the list of parameters w.r.t. which this op is to be derived
                 */
                bool set_calculate_derivative(const std::vector<Op*>&l);

                friend struct param_collector_visitor;
                friend struct toposort_visitor;
                friend struct determine_shapes_visitor;
                friend struct reset_value_set_flag;
                friend struct reset_delta_set_flag;
                friend struct swiper;

                /**
                 * show all Ops to a (constant) visitor recursively
                 */
                template<class Visitor>
                    void visit(const Visitor& v){
                        if(!v.discover(this)) return;
                        v.preorder(this);
                        BOOST_FOREACH(Op::param_t& p, m_params){
                            BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                                r->get_op()->visit(v);
                            }
                        }
                        v.postorder(this);
                    }
                /**
                 * show all Ops to a  visitor recursively
                 */
                template<class Visitor>
                    void visit(Visitor& v){
                        if(!v.discover(this)) return;
                        v.preorder(this);
                        BOOST_FOREACH(Op::param_t& p, m_params){
                            BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                                r->get_op()->visit(v);
                            }
                        }
                        v.postorder(this);
                    }


                /**
                 * user-supplied function: calculate results of this op
                 */
                virtual void fprop()=0;
                /**
                 * user-supplied function: backpropagate results of this op
                 */
                virtual void bprop()=0;

                /**
                 * virtual function: determine the shape for each result.
                 *
                 * The default works for ops with only one input:
                 * the shape of the input is simply passed to each result.
                 */
                virtual void _determine_shapes();
                private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & m_results & m_params;
                    }
        };

    class Output
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Output(){} /// for serialization
                Output(result_t& p0):Op(1,0){ 
                    add_param(0,p0);
                }
                void fprop(){
                    // simply do not reset the m_params[0] to keep the value
                }
                void bprop(){}
                void _determine_shapes(){ }
                //value_type&       data()      { return m_data; }
                const value_type& cdata() const{ return m_params[0]->value.cdata(); }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
    class Input
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                value_ptr m_data;

            public:
                Input(){} /// for serialization
                template<class T>
                    Input(const T& init):Op(0,1), m_data(new value_type(init)){  }
                void fprop(){
                    m_results[0]->push(m_data);
                    // TODO: forget m_data now?
                }
                void bprop(){}
                void _determine_shapes(){
                    m_results[0]->shape = m_data->shape();
                }
                inline value_ptr&        data_ptr()     { return m_data; }
                inline const value_ptr&  data_ptr()const{ return m_data; }

                inline value_type&       data()      { return m_data.data();  }
                inline const value_type& data() const{ return m_data.cdata(); }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_data;
                    }
        };


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
    class Pow
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

                float m_exponent;
            public:
                Pow(){} /// for serialization
                Pow(float exponent, result_t& p0):Op(1,1), m_exponent(exponent){
                    add_param(0,p0);
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();
                    value_ptr res(new value_type(inp.shape()));

                    apply_scalar_functor( *res,
                            inp, SF_POW, m_exponent);

                    r0.push(res); // 'copy' a newly created matrix

                    if(!p0.need_derivative)
                        p0.value.reset();       // forget it
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    const value_type& inp = p0.value.cdata();
                    value_ptr res(new value_type(inp.shape()));
                    apply_scalar_functor(*res,inp,SF_DPOW, m_exponent);
                    *res *= r0.delta.cdata(); // TODO: write BF_POW_TIMES functor in cuv
                    p0.push(res);
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_exponent;
                    }
        };
    class Tanh
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Tanh(){} /// for serialization
                Tanh(result_t& p0):Op(1,1){
                    add_param(0,p0);
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp = p0.value.cdata();           // original
                    value_type&      outp = p0.value.data_onlyshape();  // if detached, only allocate same size storage

                    apply_scalar_functor( outp, inp, SF_TANH);

                    r0.push(p0.value);      // 'copy' a newly created matrix
                    if(!p0.need_derivative)
                        p0.value.reset(); // forget it
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative);

                    value_type& delta = r0.delta.data(); // this is the error from above

                    const value_type& out = p0.value.cdata(); // this is the value we changed in fprop
                    value_type& res       = p0.value.data_onlyshape(); // try to overwrite this

                    apply_scalar_functor(res,out,SF_DTANH);
                    res  *=  delta;
                    p0.push(p0.value);
                    r0.delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };

    /**
     * calculates alpha * X + beta * Y, where
     * alpha, beta are scalar values and X, Y denote tensors.
     */
    class Axpby
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                float m_fact_a, m_fact_b;

            public:
                Axpby(){} /// for serialization
                Axpby(result_t& p0, result_t& p1, float fact_a=1.f, float fact_b=1.f)
                    :Op(2,1)
                     , m_fact_a(fact_a)
                     , m_fact_b(fact_b){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];

                    const value_type& inp0 = p0.value.cdata();           // original
                    const value_type& inp1 = p1.value.cdata();           // original
                    bool write_to_p0 = p0.value.unique();

                    if(r0.can_overwrite_directly()){
                        apply_binary_functor(r0.overwrite_or_add_value().data(), inp0, inp1, BF_AXPBY, m_fact_a, m_fact_b);
                    }else{
                        value_type&  outp  = write_to_p0
                            ? p0.value.data_onlyshape()
                            : p1.value.data_onlyshape();  
                        apply_binary_functor(outp, inp0, inp1, BF_AXPBY, m_fact_a, m_fact_b);
                        r0.push(write_to_p0 ? p0.value : p1.value);
                    }
                    p0.value.reset(); // forget it
                    p1.value.reset(); // forget it
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative || p1.need_derivative);

                    value_ptr delta_orig = r0.delta;
                    if(p0.need_derivative){
                        if(p0.can_add_directly()){
                            // p0 += fact_a * r0.delta
                            cuv::apply_binary_functor(p0.overwrite_or_add_value().data(),
                                    r0.delta.cdata(),
                                    BF_XPBY, m_fact_a);
                        }else if(p0.can_overwrite_directly()){
                            // p0  = fact_a * r0.delta
                            cuv::apply_scalar_functor(p0.overwrite_or_add_value().data(),
                                    r0.delta.cdata(),
                                    SF_MULT, m_fact_a);
                        }else{
                            if(!p1.need_derivative){
                                // we can only try to overwrite the current value
                                // of r0->delta if it is not needed for p1
                                delta_orig.reset();
                            }
                            // try to overwrite r0->delta
                            const value_type& inp = r0.delta.cdata();
                            value_type& outp      = r0.delta.data_onlyshape();
                            cuv::apply_scalar_functor(
                                    outp, inp, SF_MULT, m_fact_a);
                            p0.push(r0.delta);
                        }
                    }
                    if(p1.need_derivative){
                        if(p1.can_add_directly()){
                            // p1 += fact_b * r1.delta
                            cuv::apply_binary_functor(p1.overwrite_or_add_value().data(),
                                    delta_orig.cdata(),
                                    BF_XPBY, m_fact_b);
                        }else if(p1.can_overwrite_directly()){
                            // p1  = fact_b * r1.delta
                            cuv::apply_scalar_functor(p1.overwrite_or_add_value().data(),
                                    delta_orig.cdata(),
                                    SF_MULT, m_fact_b);
                        }else{
                            // try to overwrite delta_orig
                            const value_type& inp = delta_orig.cdata();
                            value_type& outp      = delta_orig.data_onlyshape();
                            cuv::apply_scalar_functor(
                                    outp, inp, SF_MULT, m_fact_b);
                            p1.push(delta_orig);
                        }
                    }
                    r0.delta.reset();
                }
                void _determine_shapes(){
                    assert(m_params[0]->shape == m_params[1]->shape);
                    m_results[0]->shape = m_params[0]->shape;
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_fact_a & m_fact_b;
                    }
        };

    class Prod
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                char m_p0t, m_p1t;
            public:
                Prod(){} /// for serialization
                Prod(result_t& p0, result_t& p1, char p0t='n', char p1t='n')
                    :Op(2,1)
                    ,m_p0t(p0t)
                    ,m_p1t(p1t)
                {
                    add_param(0,p0);
                    add_param(1,p1);
                }

                void fprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    if(r0.can_overwrite_directly()){
                        // r0 = dot(p0,p1)
                        cuv::prod(r0.overwrite_or_add_value().data(),
                                p0.value.cdata(),
                                p1.value.cdata(),
                                m_p0t, m_p1t);
                    }else if(r0.can_add_directly()){
                        // r0 += dot(p0,p1)
                        cuv::prod(r0.overwrite_or_add_value().data(),
                                p0.value.cdata(),
                                p1.value.cdata(),
                                m_p0t, m_p1t,
                                1.f,1.f);
                    }else{
                        // allocate new value *sigh*
                        value_ptr v(new value_type(r0.shape));
                        cuv::prod(*v, 
                                p0.value.cdata(),
                                p1.value.cdata(),
                                m_p0t, m_p1t);
                        r0.push(v);
                    }
                    if(!p0.need_derivative) p1.value.reset();
                    if(!p1.need_derivative) p0.value.reset();
                }
                void bprop(){
                    using namespace cuv;
                    param_t::element_type&  p0 = *m_params[0];
                    param_t::element_type&  p1 = *m_params[1];
                    result_t::element_type& r0 = *m_results[0];
                    assert(p0.need_derivative || p1.need_derivative);
                    if(p0.need_derivative){
                        const value_type& delta = r0.delta.cdata();
                        const value_type& p1v   = p1.value.cdata();
                        char p1t    = (m_p0t==m_p1t) ? 't':'n';
                        char deltat = (m_p0t=='t')   ? 't':'n';
                        if(p0.can_overwrite_directly()){
                            if(m_p0t=='n')
                                cuv::prod(p0.overwrite_or_add_value().data(),
                                        delta, p1v, deltat, p1t,1.f,0.f);
                            else
                                cuv::prod(p0.overwrite_or_add_value().data(),
                                        p1v, delta, p1t, deltat,1.f,0.f);
                        }
                        else if(p0.can_add_directly()){
                            if(m_p0t=='n')
                                cuv::prod(p0.overwrite_or_add_value().data(),
                                        delta, p1v, deltat, p1t,1.f,1.f);
                            else
                                cuv::prod(p0.overwrite_or_add_value().data(),
                                        p1v, delta, p1t, deltat,1.f,1.f);
                        }else{
                            // reallocate *sigh*
                            value_ptr v(new value_type(p0.shape));
                            if(m_p0t=='n')
                                cuv::prod(v.data(), delta, p1v,
                                        deltat, p1t,1.f,0.f);
                            else
                                cuv::prod(v.data(), p1v, delta,
                                        p1t, deltat,1.f,0.f);
                            p0.push(v);
                        }
                    }
                    if(p1.need_derivative){
                        const value_type& delta = r0.delta.cdata();
                        const value_type& p0v   = p0.value.cdata();
                        char p0t    = (m_p0t==m_p1t) ? 't':'n';
                        char deltat = (m_p1t=='t')   ? 't':'n';
                        if(p1.can_overwrite_directly()){
                            if(m_p1t=='n')
                                cuv::prod(p1.overwrite_or_add_value().data(),
                                        p0v, delta, p0t, deltat,1.f,0.f);
                            else
                                cuv::prod(p1.overwrite_or_add_value().data(),
                                        delta,p0v, deltat, p0t,1.f,0.f);
                        }
                        else if(p1.can_add_directly()){
                            if(m_p1t=='n')
                                cuv::prod(p1.overwrite_or_add_value().data(),
                                        p0v, delta, p0t,deltat,1.f,1.f);
                            else
                                cuv::prod(p1.overwrite_or_add_value().data(),
                                        delta,p0v, deltat, p0t,1.f,1.f);
                        }else{
                            // reallocate *sigh*
                            value_ptr v(new value_type(p1.shape));
                            if(m_p1t=='n')
                                cuv::prod(v.data(),
                                        p0v, delta, p0t,deltat,1.f,0.f);
                            else
                                cuv::prod(v.data(),
                                        delta,p0v, deltat,p0t,1.f,0.f);
                            p1.push(v);
                        }
                    }
                }
                void _determine_shapes(){
                    param_t&  p0 = m_params[0];
                    param_t&  p1 = m_params[1];

                    unsigned int n = m_p0t=='n' ? p0->shape[0] : p0->shape[1];
                    unsigned int m = m_p1t=='n' ? p1->shape[1] : p1->shape[0];

                    unsigned int k0 = m_p0t=='n' ? p0->shape[1] : p0->shape[0];
                    unsigned int k1 = m_p1t=='n' ? p1->shape[0] : p1->shape[1];
                    assert(k0==k1);
                    m_results[0]->shape.resize(2);
                    m_results[0]->shape[0] = n;
                    m_results[0]->shape[1] = m;
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_p0t, m_p1t;
                    }
        };

}
#endif /* __OP_HPP__ */
