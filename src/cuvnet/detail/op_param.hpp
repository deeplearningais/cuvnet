// vim:ts=4:sw=4:et:
#ifndef __OP_PARAM_HPP__
#     define __OP_PARAM_HPP__
#include <vector>
#include <boost/weak_ptr.hpp>
#include <boost/smart_ptr.hpp>
#include <cuvnet/smart_ptr.hpp>

namespace cuvnet
{
    class Op;
    namespace detail
    {
        template<class T>
            struct cmp_weak_and_raw_ptr{
                cmp_weak_and_raw_ptr(T* r):raw(r){}
                T* raw;
                bool operator()(const boost::weak_ptr<const T>& wk)const{ return wk.lock().get()==raw; }
            };
        template<class T>
            struct cmp_shared_and_raw_ptr{
                cmp_shared_and_raw_ptr(T* r):raw(r){}
                T* raw;
                bool operator()(const boost::shared_ptr<const T>& wk)const{ return wk.get()==raw; }
            };

        template<class T> struct op_result;

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
                    if(!p.delta)    return false;
                    if(!p.delta.unique())        return false;
                    if(p.delta->shape()!=p.shape) return false;
                    return true;
                }
                bool can_add_directly()const{
                    if(param_uses.size()!=1)
                        return false;
                    const op_result<T>& p = *use(0);
                    if(p.delta_set) return true;
                    return false;
                }
                void clear(){
                    BOOST_FOREACH(boost::shared_ptr<op_result<T> >& pu, param_uses){
                        pu->remove(this);
                    }
                    param_uses.clear();
                }
                void remove(op_result<T>* x){
                    param_uses.erase(
                            std::find_if(param_uses.begin(),param_uses.end(),cmp_shared_and_raw_ptr<op_result<T> >(x)),
                            param_uses.end());
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
                        // TODO: may also push to result.deltas which are not required for derivative
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
}

#endif /* __OP_PARAM_HPP__ */
