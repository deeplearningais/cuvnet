// vim:ts=4:sw=4:et:
#ifndef __OP_RESULT_HPP__
#     define __OP_RESULT_HPP__
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "cuvnet/smart_ptr.hpp"

namespace cuvnet
{
    class  Op;
    namespace detail
    {
        template<class T> struct cmp_weak_and_raw_ptr;
        template<class T> struct cmp_shared_and_raw_ptr;
        template<class T> struct op_param;

        template<class T>
            struct op_result
            : public boost::enable_shared_from_this<op_result<T> >
            {
                boost::shared_ptr<op_param<T> >       use(unsigned int i)     { return boost::shared_ptr<op_param<T> >(result_uses[i]); }
                boost::shared_ptr<const op_param<T> > use(unsigned int i)const{ return boost::shared_ptr<const op_param<T> >(result_uses[i]); }
                std::vector<boost::weak_ptr<op_param<T>> >   result_uses;
                inline unsigned int n_uses()const{return result_uses.size();}
                boost::shared_ptr<Op> op;
                std::vector<unsigned int>      shape;
                bool                           need_result;
                //cow_ptr<T>                     value;
                cow_ptr<T>                     delta;
                bool                           delta_set;
                unsigned int                   result_number;
                typename std::vector<boost::weak_ptr<op_param<T> > >::iterator m_single_result;

                boost::shared_ptr<Op> get_op(){ return op; }
                op_result():need_result(false),delta_set(false),m_single_result(result_uses.end()){}
                void determine_single_results(){
                    m_single_result = result_uses.end();
                    for(typename std::vector<boost::weak_ptr<op_param<T> > >::iterator it=result_uses.begin();
                            it != result_uses.end();
                            ++it){
                        if(m_single_result != result_uses.end())
                        {
                            m_single_result = result_uses.end();
                            return;
                        }
                        m_single_result = it;
                    }
                }
                bool can_overwrite_directly()const{
                    if(m_single_result == result_uses.end())
                        return false;
                    const op_param<T>& p = * m_single_result->lock();
                    if(p.value_set) return false;
                    if(!p.value)    return false;
                    if(!p.value.unique())        return false;
#ifndef NDEBUG
                    cuvAssert(p.value->shape() == p.shape);
                    //if(p.value->shape()!=p.shape) return false;
#endif
                    return true;
                }
                bool can_add_directly()const{
                    if(m_single_result == result_uses.end())
                        return false;
                    const op_param<T>& p = * m_single_result->lock();
                    if(!p.value) return false;
#ifndef NDEBUG
                    cuvAssert(p.value->shape() == p.shape);
                    //if(p.value->shape()!=p.shape) return false;
#endif
                    if(p.value_set) return true;
                    return false;
                }
                void clear(){
                    BOOST_FOREACH(boost::weak_ptr<op_param<T> >& pu, result_uses){
                        pu.lock()->remove(this);
                    }
                    result_uses.clear();
                }
                void remove(op_param<T>* x){
                    result_uses.erase(
                            std::remove_if(result_uses.begin(),result_uses.end(),cmp_weak_and_raw_ptr<op_param<T> >(x)),
                            result_uses.end());
                    if(result_uses.empty())
                        op.reset(); // forget op, so that it can be destroyed if needed!
                }
                /**
                 * get the value to write at directly, also sets value_set for convenience.
                 *
                 */
                cow_ptr<T>& overwrite_or_add_value(){
                    //assert(m_single_result != result_uses.end());
                    m_single_result->lock()->value_set = true;
                    return m_single_result->lock()->value;
                }
                void push(const cow_ptr<T>& v){
                    //assert(!can_overwrite_directly());
                    //assert(!can_add_directly());
                    for (unsigned int i = 0; i < result_uses.size(); ++i)
                    {
                        op_param<T>& dst    = *use(i);
                        if(!dst.get_op()->need_result())
                            continue;
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
                        ar & op & result_uses & result_number;
                    }
            };

    }
}
#endif /* __OP_RESULT_HPP__ */
