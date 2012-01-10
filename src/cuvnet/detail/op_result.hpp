// vim:ts=4:sw=4:et:
#ifndef __OP_RESULT_HPP__
#     define __OP_RESULT_HPP__

namespace cuvnet
{
    class  Op;
    namespace detail
    {
        template<class T> struct cmp_weak_and_raw_ptr;
        template<class T> struct cmp_shared_and_raw_ptr;
        template<class T> struct op_param;

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
                unsigned int                   result_number;
                boost::shared_ptr<Op> get_op(){ return op; }
                bool want_result()const { return result_uses.size() > 0; }
                bool can_overwrite_directly()const{
                    if(result_uses.size()!=1)
                        return false;
                    const op_param<T>& p = *use(0);
                    if(p.value_set) return false;
                    if(!p.value)    return false;
                    if(!p.value.unique())        return false;
                    if(p.value->shape()!=p.shape) return false;
                    return true;
                }
                bool can_add_directly()const{
                    if(result_uses.size()!=1)
                        return false;
                    const op_param<T>& p = *use(0);
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
                            std::find_if(result_uses.begin(),result_uses.end(),cmp_weak_and_raw_ptr<op_param<T> >(x)),
                            result_uses.end());
                    if(result_uses.empty())
                        op.reset(); // forget op, so that it can be destroyed if needed!
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
                    for (unsigned int i = 0; i < result_uses.size(); ++i)
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
                        ar & op & result_uses & result_number;
                    }
            };

    }
}
#endif /* __OP_RESULT_HPP__ */
