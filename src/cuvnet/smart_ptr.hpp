// vim:ts=4:sw=4:et
#ifndef __CUVNET_SMART_PTR_HPP__
#     define __CUVNET_SMART_PTR_HPP__
#include <ostream>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/make_shared.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <cuv/basics/tensor.hpp>

namespace cuvnet
{
    template <class T>
        class cow_ptr
        {
            public:
                typedef boost::shared_ptr<T> ref_ptr;
                cow_ptr( ){}
                cow_ptr( const cow_ptr &cpy ) : m_ptr(cpy.m_ptr){}
                explicit cow_ptr( const ref_ptr &cpy ) : m_ptr(cpy){}
                explicit cow_ptr(       T*       cpy ) : m_ptr(cpy){}

                inline void reset( T* cpy )   { m_ptr.reset(cpy);  }
                inline void reset(        )   { m_ptr.reset();     }
                void detach(){
                    T* tmp = m_ptr.get();
                    if( ! (tmp==0 || m_ptr.unique()))
                        m_ptr.reset(new T(*tmp));
                }
                void detach_onlyshape(){
                    T* tmp = m_ptr.get();
                    if( ! (tmp==0 || m_ptr.unique()))
                        m_ptr.reset(new T(tmp->shape()));
                }
                cow_ptr& operator=(const cow_ptr& o){ m_ptr = o.m_ptr; return *this;}
                cow_ptr& operator=(const T& t){ 
                    if(!m_ptr) 
                        m_ptr.reset(new T(t)); //  nothing was set--> store new obj
                    else if(m_ptr.unique())
                        *m_ptr = t;            //  sole owner changes ptr-->overwrite
                    else{
                        m_ptr.reset(new T(t)); // multiple owners, change only this copy
                    }
                    
                    return *this;
                }

                size_t ptr(){ return (size_t) m_ptr.get(); }

                // const versions
                const T&       data() const { return *m_ptr; }
                const T&      cdata() const { return *m_ptr; }
                const T& operator* () const { return *m_ptr; }
                const T* operator->() const { return  m_ptr.operator->(); }
                operator const T& ()const{ return *m_ptr; }
                bool     operator!()const   { return !m_ptr; }

                // non-const versions
                T&        data_onlyshape()  { detach_onlyshape(); return *m_ptr; }
                T&        data()         { detach(); return *m_ptr; }
                const T& cdata()         {           return *m_ptr; }
                T& operator* ()          { detach(); return *m_ptr; }
                T* operator->()          { detach(); return  m_ptr.operator->(); }
                operator T& ()  { detach(); return *m_ptr; }

                template<class U>
                friend std::ostream& operator<< (std::ostream &o, const cow_ptr<U>&);

            private:
                ref_ptr m_ptr;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & m_ptr;
                    }
        };

        template<class T>
        std::ostream& 
        operator<< (std::ostream &o, const cow_ptr<T>& p){
            o << (T)p; return o;
        }

    template<class T>
        struct locked_obj
        {
            std::vector<void*> locks;
            std::auto_ptr<T> ptr;
            bool flag;
            locked_obj(T*t):ptr(t),flag(false){}
            locked_obj(T&t):ptr(new T(t)),flag(false){}
        };
        /** 
         * copy eliminiation pointer
         */
    template <class T>
        class ce_ptr
        {
            public:
                typedef locked_obj<T>  ref_t;
                typedef boost::shared_ptr<ref_t> ref_ptr;
                ce_ptr( ){}
                ce_ptr( const ce_ptr &cpy ) : m_ptr(cpy.m_ptr){}
                explicit ce_ptr( const ref_ptr &cpy ) : m_ptr(cpy){}
                explicit ce_ptr(       T*       cpy ) : m_ptr(new ref_t(cpy)){}

                /**
                 * determines whether we are the sole owner of this pointer
                 *
                 * @param p lockers equal to p are ignored
                 */
                bool writable(void* p=NULL){
                    std::vector<void*>& locks = m_ptr->locks;
                    if(locks.end() 
                            != std::find_if(locks.begin(), locks.end(),
                                std::bind2nd(std::not_equal_to<void*>(), p)))
                        return false; // somebody else has locked this
                    return true;
                }
                bool locked_by(void* p){
                    std::vector<void*>& locks = m_ptr->locks;
                    return locks.end()
                            != std::find_if(locks.begin(), locks.end(),
                                std::bind2nd(std::equal_to<void*>(), p));
                }
                operator bool ()const{
                    return !(m_ptr.get()==NULL||m_ptr->ptr.get()==NULL);
                }
                bool operator!()const{
                    return (m_ptr.get()==NULL || m_ptr->ptr.get()==NULL);
                }
                void reset(T* t){
                    m_ptr.reset(new ref_t(t)); //  nothing was set--> store new obj
                }
                bool flagged()           const{return m_ptr->flag;}
                void flag(bool b=true)        {m_ptr->flag=b;}
                ce_ptr& operator=(const ce_ptr& o){ m_ptr = o.m_ptr; return *this;}

                size_t ptr()const{ return (size_t) m_ptr.get(); }

                // const versions (no need to detach)
                const T&       data() const { return *m_ptr->ptr; }
                const T&      cdata() const { return *m_ptr->ptr; }
                const T& operator* () const { return *m_ptr->ptr; }
                const T* operator->() const { return m_ptr->ptr.operator->(); }
                operator const T& ()const{ return *m_ptr->ptr; }

                // non-const versions (detach if necessary)
                T&        data(void* p)  { 
                    if(!writable(p)){detach(p);}
                    else{assert(locked_by(p));}
                    return *m_ptr->ptr; 
                }
                const T& cdata()         {            return *m_ptr->ptr; }

                template<class U>
                friend std::ostream& operator<< (std::ostream &o, const ce_ptr<U>&);

                ce_ptr&   lock(void* p){ m_ptr->locks.push_back(p);  return *this;}
                ce_ptr& unlock(void* p){ 
                    std::vector<void*>& locks = m_ptr->locks;
                    locks.erase(
                            std::remove_if(locks.begin(),locks.end(),
                                std::bind2nd(std::equal_to<void*>(), p)),
                            locks.end());
                    return *this;
                }

            private:
                ref_ptr m_ptr;

            private:
                void detach(void* p){
                    ref_t* tmp = m_ptr.get();
                    if( ! (tmp==0 || m_ptr.unique())){
                        unlock(p); // previous object does not depend on p anymore
                        T* tmpT = tmp->ptr.get();
                        if(tmpT)
                            m_ptr.reset(new ref_t(new T(*tmpT))); // no locks here!
                    }
                }
        };

        template<class T>
        std::ostream& 
        operator<< (std::ostream &o, const ce_ptr<T>& p){
            o << (T)p; return o;
        }

}
#endif /* __CUVNET_SMART_PTR_HPP__ */
