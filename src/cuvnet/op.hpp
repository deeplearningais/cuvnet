#ifndef __OP_HPP__
#     define __OP_HPP__
#include <list>
#include <map>
#include <boost/weak_ptr.hpp>
#include <boost/foreach.hpp>
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
                cow_ptr<T>& overwrite_or_add_value(){
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
                cow_ptr<T>                     delta;
                bool                           value_set;

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
                cow_ptr<T>& overwrite_or_add_value(){
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
            };
    }


    class Op
        : public boost::enable_shared_from_this<Op>{
            public:
            typedef matrix value_type;
            typedef cow_ptr<value_type>                 value_ptr;
            typedef boost::shared_ptr<Op>               op_ptr;
            typedef boost::shared_ptr<detail::op_param<value_type> >        param_t;
            typedef boost::shared_ptr<detail::op_result<value_type> >      result_t;
            protected:
                std::vector<param_t>  m_params;
                std::vector<result_t> m_results;

            public:
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
			inline bool set_calculate_derivative(const std::list<Op*>&l){
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
                std::list<Op*>     plist;
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
                std::vector<Op*>     plist;
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
            /*
             *[**
             * * weak pointers to the ops cannot be created in constructors, we
             * * need to post-hoc wire them.  another alternative is to make
             * * constructors private as in:
             * * http://stackoverflow.com/questions/4598986/weak-pointer-to-this-in-constructor
             * * but this would change the C++-y interface.
             * *]
             *struct wire_visitor : public op_visitor_adaptor{
             *    inline bool preorder(Op* o)const{
             *        BOOST_FOREACH(Op::result_t& p, o->m_results){
             *            p->op = o->shared_from_this();
             *        }
             *        BOOST_FOREACH(Op::param_t& p, o->m_params){
             *            p->op = o->shared_from_this();
             *        }
             *    }
             *};
             */

            /**
             * determine shapes recursively
             */
            struct determine_shapes_visitor :public op_visitor_adaptor{
                inline void postorder(Op* o)const{
                    o->_determine_shapes();
                    // push from result to result-users
                    BOOST_FOREACH(Op::result_t& r, o->m_results){
                        for(unsigned int i=0;i<r->result_uses.size();i++){
                            r->use(i)->shape = r->shape;
                        }
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
        };

    class Output
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
                Output(result_t& p0):Op(1,0){ 
                    add_param(0,p0);
                }
                void fprop(){
                    // simply store the inputs
                    m_data = m_params[0]->value;
                    m_params[0]->value.reset();
                }
                void bprop(){}
                void _determine_shapes(){ }
                //value_type&       data()      { return m_data; }
                const value_type& data() const{ return m_data; }
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
                template<class T>
                Input(const T& init):Op(0,1), m_data(new value_type(init)){  }
                void fprop(){
                    m_results[0]->push(m_data);
                }
                void bprop(){}
                void _determine_shapes(){
                    m_results[0]->shape = m_data->shape();
                }
                value_type&       data()      { return m_data; }
                const value_type& data() const{ return m_data; }
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
        };
    class Pow
        : public Op{
            typedef Op::value_type    value_type;
            typedef Op::op_ptr        op_ptr;
            typedef Op::value_ptr     value_ptr;
            typedef Op::param_t       param_t;
            typedef Op::result_t      result_t;

            float m_exponent;
            public:
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
                p0->push(res);
            }
        };
    class Tanh
        : public Op{
            typedef Op::value_type    value_type;
            typedef Op::op_ptr        op_ptr;
            typedef Op::value_ptr     value_ptr;
            typedef Op::param_t       param_t;
            typedef Op::result_t      result_t;

            public:
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
                p0->value.reset(); // forget it
            }
            void bprop(){
                using namespace cuv;
                param_t&  p0 = m_params[0];
                result_t& r0 = m_results[0];
                assert(p0->need_derivative);

                value_type& delta = r0->delta.data();

                apply_scalar_functor(delta,delta,SF_DTANH);
                r0->push(r0->delta);
                r0->delta.reset();
            }
        };
}
#endif /* __OP_HPP__ */
