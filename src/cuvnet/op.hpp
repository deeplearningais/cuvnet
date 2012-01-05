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
                boost::weak_ptr<Op> op;
                //cow_ptr<T>                     value;
                cow_ptr<T>                     delta;
                bool                           delta_set;
                boost::shared_ptr<Op> get_op(){ return boost::shared_ptr<Op>(op); }
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
                    assert(!can_overwrite_directly());
                    assert(!can_add_directly());
                    for (int i = 0; i < result_uses.size(); ++i)
                    {
                        op_param<T>& dst    = *use(i);
                        if(dst.value_set)
                            *dst.value     += v.cdata();
                        else
                            dst.value       = v;
                    }
                }
            };
        template<class T>
            struct op_param{
                boost::shared_ptr<op_result<T> > use(unsigned int i){ return param_uses[i]; }
                boost::shared_ptr<const op_result<T> > use(unsigned int i)const{ return param_uses[i]; }
                std::vector<boost::shared_ptr<op_result<T> > >     param_uses;
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
                    assert(!can_overwrite_directly());
                    assert(!can_add_directly());
                    for (int i = 0; i < param_uses.size(); ++i)
                    {
                        op_param<T>& dst    =use(i); 
                        if(dst.value_set)
                            dst.value.data() += v.cdata();
                        else
                            dst.value       = v;
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
            void set_n_params(unsigned int n){ 
                m_params.resize(n); 
                for(int i=0;i<n;i++)
                    m_params[i]->op = shared_from_this();
            }
            void set_n_results(unsigned int n){ 
                m_results.resize(n); 
                for(int i=0;i<n;i++)
                    m_results[i]->op = shared_from_this();
            }
            unsigned int get_n_params(){ return m_params.size(); }
            unsigned int get_n_results(){ return m_results.size(); }

			/**
			 * calculate recursively what needs to be calculated to
			 * derive this operator w.r.t. a set of parameters.
			 *
			 * The results are stored in the function itself.
			 *
			 * @param l the list of parameters w.r.t. which this op is to be derived
			 */
            template<class T>
			inline bool set_calculate_derivative(const std::list<T>&l){
                if(l.end() != std::find(l.begin(),l.end(), 
                            std::bind2nd(
                                std::equal_to<T>(),
                                (T)shared_from_this()))){
                    assert(m_params.size()==0); // this should be a "scalar"
                    return true;
                }
                bool need_calc_derivative = false;
                BOOST_FOREACH(param_t& p, m_params){
                    bool derive_wrt_p = false;
                    BOOST_FOREACH(detail::op_result<value_type>* r, p->param_uses){
                        derive_wrt_p |= r->get_op()->set_calculate_derivative(l);
                    }
                    p->need_derivative = derive_wrt_p;
                    need_calc_derivative |= derive_wrt_p;
                }
                return need_calc_derivative;
			}

            struct param_collector{
                std::list<Op*>     plist;
                std::map<Op*,bool> visited;
                inline bool preorder(Op* o){
                    if(visited.find(o)!=visited.end())
                        return true;
                    visited[o]=true;
                    if(o->get_n_params()==0)
                        plist.push_back(o);
                    return true;
                }
                inline bool postorder(Op* o){
                    return true;
                }
            };
            template<class Visitor>
            bool visit(Visitor& v){
                if(!v.preorder(this)) return false;
                BOOST_FOREACH(param_t& p, m_params){
                    BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                        if(!r->get_op()->visit(v))
                            return false;
                    }
                }
                return v.postorder(this);
            }

            virtual void fprop()=0;
            virtual void bprop()=0;
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
            Identity():Op(1,1){}

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
                p0->value.reset(); // don't need that for backprop etc.
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

    class Input
        : public Op{
            typedef Op::value_type    value_type;
            typedef Op::op_ptr        op_ptr;
            typedef Op::value_ptr     value_ptr;
            typedef Op::param_t       param_t;
            typedef Op::result_t      result_t;
            public:
            Input():Op(0,1){ }
        };

    class Pow
        : public Op{
            typedef Op::value_type    value_type;
            typedef Op::op_ptr        op_ptr;
            typedef Op::value_ptr     value_ptr;
            typedef Op::param_t       param_t;
            typedef Op::result_t      result_t;

            public:
            void add_param(unsigned int idx, result_t& p){
                m_params[idx]->param_uses.push_back(p);
                p->result_uses.push_back(m_params[idx]);
            }
            Pow(float exponent, result_t& p0):Op(1,1){
                add_param(0,p0);
            }

            void fprop(){
            }
            void bprop(){
            }
        };
}
#endif /* __OP_HPP__ */
