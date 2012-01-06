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

namespace cuvnet
{
    typedef char param_name_t;
    class Op;


    namespace detail{

        template<class T> class op_param;

        template<class T>
            struct op_result{
                boost::shared_ptr<op_param<T> >       use(unsigned int i)     { return boost::shared_ptr<op_param<T> >(result_uses[i]); }
                boost::shared_ptr<const op_param<T> > use(unsigned int i)const{ return boost::shared_ptr<const op_param<T> >(result_uses[i]); }
                std::vector<boost::weak_ptr<op_param<T>> >   result_uses;
                boost::shared_ptr<Op> op;
                std::vector<unsigned int>      shape;
                //cow_ptr<T>                     value;
                cow_ptr<T>                     delta;
                bool                           delta_set;
                boost::shared_ptr<Op> get_op(){ return op; }
                bool want_result()const { return result_uses.size() > 0; }
                bool can_overwrite_directly()const{
                    if(result_uses.size()!=1)
                        return false;
                    const op_param<T>& p = *use(0);
                    if(p.value_set) return false;
                    return true;
                }
                bool can_add_directly()const{
                    if(result_uses.size()!=1)
                        return false;
                    const op_param<T>& p = *use(0);
                    if(p.value_set) return true;
                    return false;
                }
                /**
                 * get the value to write at directly, also sets value_set for convenience
                 *
                 */
                cow_ptr<T>& overwrite_or_add_value(){
                    use(0)->value_set = true;
                    return use(0)->value;
                }
                void push(const cow_ptr<T>& v){
                    //assert(!can_overwrite_directly());
                    //assert(!can_add_directly());
                    for (int i = 0; i < result_uses.size(); ++i)
                    {
                        op_param<T>& dst    = *use(i);
                        if(dst.value_set)
                            *dst.value     += v.cdata();
                        else{
                            dst.value       = v;
                            dst.value_set   = true;
                        }
                    }
                }

                private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & op & result_uses;
                    }
            };
        template<class T>
            struct op_param{
                boost::shared_ptr<op_result<T> >&       use(unsigned int i){ return param_uses[i]; }
                boost::shared_ptr<const op_result<T> >  use(unsigned int i)const{ return param_uses[i]; }
                std::vector<boost::shared_ptr<op_result<T> > >     param_uses;
                std::vector<unsigned int>      shape;
                bool                           need_derivative;
                boost::weak_ptr<Op>            op;
                cow_ptr<T>                     value;
                bool                           value_set;
                //cow_ptr<T>                     delta;

                boost::shared_ptr<Op> get_op(){ return boost::shared_ptr<Op>(op); }
                bool can_overwrite_directly()const{
                    if(param_uses.size()!=1)
                        return false;
                    const op_result<T>& p = *use(0);
                    if(p.delta_set) return false;
                    return true;
                }
                bool can_add_directly()const{
                    if(param_uses.size()!=1)
                        return false;
                    const op_result<T>& p = *use(0);
                    if(p.delta_set) return true;
                    return false;
                }
                /**
                 * get the delta to write at directly, also sets delta_set for convenience
                 *
                 */
                cow_ptr<T>& overwrite_or_add_value(){
                    use(0)->delta_set = true;
                    return use(0)->delta;
                }
                void push(const cow_ptr<T>& v){
                    //assert(!can_overwrite_directly());
                    //assert(!can_add_directly());
                    for (int i = 0; i < param_uses.size(); ++i)
                    {
                        op_result<T>& dst    = *use(i); 
                        if(dst.delta_set)
                            dst.delta.data() += v.cdata();
                        else{
                            dst.delta       = v;
                            dst.delta_set   = true;
                        }
                    }
                }
                private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & op & param_uses;
                    }
            };
    }


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
                Op(){
                }
                Op(unsigned int n_params, unsigned int n_results){
                    set_n_params(n_params);
                    set_n_results(n_results);
                }

                /** 
                 * returns a reference to the i-th result
                 *
                 * @note the included check is inefficient but avoids making constructor private
                 *       since we cannot determine `shared_from_this' in
                 *       the constructor when we construct objects in m_results.
                 *       I assume that result() will be primarily used to construct
                 *       functions, which is not that often.
                 */
                result_t&       result(const unsigned int i=0){
                    if(!m_results[i]->op)
                        m_results[i]->op = shared_from_this();
                    return m_results[i];
                }
                /** 
                 * returns a reference to the i-th parameter
                 */
                param_t&       param(const unsigned int i=0){
                    return m_params[i];
                }
                void set_n_params(unsigned int n){ 
                    m_params.resize(n); 
                    for(int i=0;i<n;i++){
                        m_params[i].reset(new detail::op_param<value_type>());
                    }
                }
                void set_n_results(unsigned int n){ 
                    m_results.resize(n); 
                    for(int i=0;i<n;i++){
                        m_results[i].reset(new detail::op_result<value_type>());
                    }
                }
                unsigned int get_n_params(){ return m_params.size(); }
                unsigned int get_n_results(){ return m_results.size(); }
                void add_param(unsigned int idx, result_t& p){
                    param(idx)->param_uses.push_back(p);
                    p->result_uses.push_back(param(idx));
                }

                /**
                 * calculate recursively what needs to be calculated to
                 * derive this operator w.r.t. a set of parameters.
                 *
                 * The results are stored in the function itself.
                 *
                 * @param l the list of parameters w.r.t. which this op is to be derived
                 */
                inline bool set_calculate_derivative(const std::vector<Op*>&l){
                    if(l.end() != std::find(l.begin(),l.end(), this)){
                        assert(m_params.size()==0); // this should be a "scalar"
                        return true;
                    }
                    bool need_calc_derivative = false;
                    BOOST_FOREACH(param_t& p, m_params){
                        bool derive_wrt_p = false;
                        BOOST_FOREACH(Op::result_t& r, p->param_uses){
                            derive_wrt_p |= r->get_op()->set_calculate_derivative(l);
                        }
                        p->need_derivative = derive_wrt_p;
                        need_calc_derivative |= derive_wrt_p;
                    }
                    return need_calc_derivative;
                }

                /**
                 * helper class to create visitors (you can derive from this so
                 * that you e.g. only need to implement one method)
                 */
                struct op_visitor_adaptor{
                    inline bool discover(Op* o)const{ return true; }
                    inline void preorder(Op* o)const{ ; }
                    inline void postorder(Op* o)const{ ; }
                };
                /**
                 * collect all no-input ops in a list
                 */
                struct param_collector_visitor : public op_visitor_adaptor{
                    typedef std::vector<Op*> container_type;
                    container_type     plist;
                    std::map<Op*,bool> visited;
                    inline bool discover(Op* o){
                        if(visited.find(o)!=visited.end())
                            return false;
                        visited[o]=true;
                        return true;
                    }
                    inline void preorder(Op* o){
                        if(o->get_n_params()==0)
                            plist.push_back(o);
                    }
                };
                /**
                 * collect all ops in a list in topological order
                 */
                struct toposort_visitor : public op_visitor_adaptor{
                    typedef std::vector<Op*> container_type;
                    container_type     plist;
                    std::map<Op*,bool> visited;
                    bool               deriv_only;
                    toposort_visitor(bool deriv):deriv_only(deriv){}
                    inline bool discover(Op* o){
                        if(visited.find(o)!=visited.end()) return false;
                        if(deriv_only){

                            if(o->m_params.size()==0) // input
                                return true;
                            for (int i = 0; i < o->m_params.size(); ++i)
                            {
                                // at least one parameter should have this set
                                if(o->m_params[i]->need_derivative){
                                    visited[o] = true;
                                    return true;
                                }
                            }
                        }
                        return false;
                    }
                    inline void postorder(Op* o){
                        plist.push_back(o);
                    }
                };

                /**
                 * determine shapes recursively
                 */
                struct determine_shapes_visitor :public op_visitor_adaptor{
                    bool deriv_only;
                    determine_shapes_visitor(bool deriv=false):deriv_only(deriv){}
                    inline void postorder(Op* o)const{
                        o->_determine_shapes();
                        // push from result to result-users
                        BOOST_FOREACH(Op::result_t& r, o->m_results){
                            for(unsigned int i=0;i<r->result_uses.size();i++){
                                if(!deriv_only || r->use(i)->need_derivative)
                                    r->use(i)->shape = r->shape;
                            }
                        }
                    }
                };

                /**
                 * reset the `delta_set' flag before a bprop-pass
                 */
                struct reset_delta_set_flag : public op_visitor_adaptor{
                    inline void preorder(Op*o)const{
                        BOOST_FOREACH(Op::result_t& r, o->m_results){
                            r->delta_set = false;
                        }
                    }
                };
                /**
                 * reset the `value_set' flag before a fprop-pass
                 */
                struct reset_value_set_flag : public op_visitor_adaptor{
                    inline void preorder(Op*o)const{
                        BOOST_FOREACH(Op::param_t& r, o->m_params){
                            r->value_set = false;
                        }
                    }
                };

                /**
                 * show all Ops to a (constant) visitor recursively
                 */
                template<class Visitor>
                    bool visit(const Visitor& v){
                        if(!v.discover(this)) return true;
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
                    bool visit(Visitor& v){
                        if(!v.discover(this)) return true;
                        v.preorder(this);
                        BOOST_FOREACH(Op::param_t& p, m_params){
                            BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                                !r->get_op()->visit(v);
                            }
                        }
                        v.postorder(this);
                    }

                /**
                 * does a recursive forward/backward pass w.r.t. 
                 * requested parameters.
                 *
                 * To do passes, the structure of the operator is
                 * sorted topologically once (in the constructor).
                 * Consecutive calles should therefore be done with
                 * the same `swiper' object.
                 */
                struct swiper{
                    toposort_visitor m_topo;
                    /**
                     * constructor
                     *
                     * @param op      the operator to do swipes on
                     * @param deriv   whether to only do passes w.r.t. the named parameters
                     * @param paramlist the list of parameters w.r.t. which do swipes
                     */
                    swiper(Op& op, bool deriv, const param_collector_visitor::container_type& paramlist)
                        :m_topo(deriv){
                            op.set_calculate_derivative(paramlist);
                            op.visit(m_topo);
                            op.visit(Op::determine_shapes_visitor());
                        }
                    /**
                     * does recursive forward pass on op
                     */
                    void fprop(){
                        BOOST_FOREACH(Op* o, m_topo.plist){
                            BOOST_FOREACH(Op::result_t& r, o->m_results){
                                BOOST_FOREACH(Op::weak_param_t p, r->result_uses){
                                    p.lock()->value_set = false;
                                }
                            }
                        }
                        BOOST_FOREACH(Op* o, m_topo.plist){
                            o->fprop();
                        }
                    }
                    /**
                     * does recursive backward pass on op
                     */
                    void bprop(){
                        BOOST_FOREACH(Op* o, m_topo.plist){
                            BOOST_FOREACH(Op::param_t& p, o->m_params){
                                BOOST_FOREACH(Op::result_t& r, p->param_uses){
                                    r->delta_set = false;
                                }
                            }
                        }
                        BOOST_REVERSE_FOREACH(Op* o, m_topo.plist){
                            o->bprop();
                        }
                    }
                };

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
                virtual void _determine_shapes(){
                    assert(m_params.size()==1);
                    BOOST_FOREACH(result_t& r, m_results){
                        r->shape = m_params[0]->shape;
                    }
                }
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
                value_ptr&        data_ptr()     { return m_data; }
                const value_ptr&  data_ptr()const{ return m_data; }

                value_type&       data()      { return m_data.data();  }
                const value_type& data() const{ return m_data.cdata(); }
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
                    param_t&  p0 = m_params[0];
                    result_t& r0 = m_results[0];

                    if(r0->can_overwrite_directly()){
                        value_ptr& ptr = r0->overwrite_or_add_value();
                        ptr            = p0->value;
                    }else if(r0->can_add_directly()){
                        value_ptr& ptr = r0->overwrite_or_add_value();
                        *ptr          += p0->value.cdata();
                    }else{
                        r0->push(p0->value); // 'copy' a newly created matrix
                    }
                    p0->value.reset();       // don't need that for backprop etc.
                }
                void bprop(){
                    // identity
                    using namespace cuv;
                    param_t&  p0 = m_params[0];
                    result_t& r0 = m_results[0];

                    if(p0->can_overwrite_directly()){
                        value_ptr& ptr = p0->overwrite_or_add_value();
                        ptr            = r0->delta;
                    }else if(p0->can_add_directly()){
                        value_ptr& ptr = p0->overwrite_or_add_value();
                        *ptr          += r0->delta.cdata();
                    }else{
                        r0->push(p0->value); // 'copy' a newly created matrix
                    }
                    r0->delta.reset();
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
                    param_t&  p0 = m_params[0];
                    result_t& r0 = m_results[0];

                    const value_type& inp = p0->value.cdata();
                    value_ptr res(new value_type(inp.shape()));

                    apply_scalar_functor( *res,
                            inp, SF_POW, m_exponent);

                    r0->push(res); // 'copy' a newly created matrix

                    if(!p0->need_derivative)
                        p0->value.reset();       // forget it
                }
                void bprop(){
                    using namespace cuv;
                    param_t&  p0 = m_params[0];
                    result_t& r0 = m_results[0];
                    assert(p0->need_derivative);

                    const value_type& inp = p0->value.cdata();
                    value_ptr res(new value_type(inp.shape()));
                    apply_scalar_functor(*res,inp,SF_DPOW, m_exponent);
                    *res *= r0->delta.cdata(); // TODO: write BF_POW_TIMES functor in cuv
                    p0->push(res);
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
                    param_t&  p0 = m_params[0];
                    result_t& r0 = m_results[0];

                    const value_type& inp = p0->value.cdata();           // original
                    value_type&      outp = p0->value.data_onlyshape();  // if detached, only allocate same size storage

                    apply_scalar_functor( outp, inp, SF_TANH);

                    r0->push(p0->value);      // 'copy' a newly created matrix
                    if(!p0->need_derivative)
                        p0->value.reset(); // forget it
                }
                void bprop(){
                    using namespace cuv;
                    param_t&  p0 = m_params[0];
                    result_t& r0 = m_results[0];
                    assert(p0->need_derivative);

                    value_type& delta = r0->delta.data(); // this is the error from above

                    const value_type& out = p0->value.cdata(); // this is the value we changed in fprop
                    value_type& res       = p0->value.data_onlyshape(); // try to overwrite this

                    apply_scalar_functor(res,out,SF_DTANH);
                    res  *=  delta;
                    p0->push(p0->value);
                    r0->delta.reset();
                }
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
}
#endif /* __OP_HPP__ */
