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
    namespace detail {
        template<class T>
            struct cow_ptr_traits{
                static T* clone(const T& t){ return new T(t); }
                static void check(const T* t){}
            };
        template<class V, class M, class L>
            struct cow_ptr_traits<cuv::tensor<V,M,L> >{
                typedef cuv::tensor<V,M,L> T;
                static T* clone(const T& t){ return new T(t,cuv::linear_memory_tag()); }
                static void check(const T* t){
                    if(t){
                        cuvAssert(!cuv::has_nan(*t));
                        cuvAssert(!cuv::has_inf(*t));
                    }
                }
            };
    }

    /**
     * A copy-on-write pointer.
     *
     * The idea is that we can spread the data stored in this pointer to many
     * consumers which may only want to read it. As they are processing the
     * output in sequence, they should reset `their' pointer once they don't
     * need it anymore.
     *
     * The last (or only!) one of the consumers can then safely overwrite the data.
     * If somebody wants to modify the content while it is still required by
     * others, the content is copied and the pointer detached from the original
     * data.
     */
    template <class T>
        class cow_ptr
        {
            public:
                /// type of the contained data -- a shared pointer!
                typedef boost::shared_ptr<T> ref_ptr;
                /// constructor (does nothing)
                cow_ptr( ){}
                /// copy constructor (cheap)
                cow_ptr( const cow_ptr &cpy ) : m_ptr(cpy.m_ptr){}
                /// create from shared_ptr (cheap)
                explicit cow_ptr( const ref_ptr &cpy ) : m_ptr(cpy){}
                /// create from raw pointer (cheap, takes ownership)
                explicit cow_ptr(       T*       cpy ) : m_ptr(cpy){}

                /// reset using raw pointer (takes ownership)
                inline void reset( T* cpy )     { m_ptr.reset(cpy);      }
                /// release pointer (may not destroy if there are other copies)
                inline void reset(        )     { m_ptr.reset();         }
                /// true if no copies exist
                inline bool unique(       )const{ return m_ptr.unique(); }
                /// expensive if copies exist: clones contained object.
                void detach(){
                    T* tmp = m_ptr.get();
                    if( ! (tmp==0 || m_ptr.unique()))
                        m_ptr.reset(detail::cow_ptr_traits<T>::clone(*tmp));// force copying!
                }
                /// creates new object with same shape (mainly for cuv::tensor objects)
                void detach_onlyshape(){
                    T* tmp = m_ptr.get();
                    if( ! (tmp==0 || m_ptr.unique())){
                        m_ptr.reset(new T(tmp->shape()));
                    }
                }
                /// assignment operator (other cow_ptr)
                cow_ptr& operator=(const cow_ptr& o){ m_ptr = o.m_ptr; return *this;}
                /// assignment operator (reference, copies! --> expensive)
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

                /// return internal pointer. Can be used to get around restrictions, use w/ care!
                T* ptr(){ return  m_ptr.get(); }

                /// const versions
                /// @{
                const T&       data() const { return *m_ptr; } ///< const contained data, never detaches
                const T&      cdata() const { return *m_ptr; } ///< const contained data, never detaches
                const T& operator* () const { return *m_ptr; } ///< const contained data, never detaches
                const T* operator->() const { return  m_ptr.operator->(); } ///< const contained data, never detaches
                operator const T& ()const{ return *m_ptr; } ///< conversion to const contained data, never detaches
                bool     operator!()const   { return !m_ptr; } ///< true if pointing to NULL
                /// @}

                /// non-const versions
                /// @{
                T&        data_onlyshape()  { detach_onlyshape(); return *m_ptr; } ///< contained data, detaches if needed
                T&        data()         { detach(); return *m_ptr; } ///< contained data, detaches if needed
                const T& cdata()         {           return *m_ptr; } ///< contained data, detaches if needed
                T& operator* ()          { detach(); return *m_ptr; } ///< contained data, detaches if needed
                T* operator->()          { detach(); return  m_ptr.operator->(); } ///< contained data, detaches if needed
                operator T& ()  { detach(); return *m_ptr; } ///< conversion to contained data, detaches if needed
                /// @}

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
}
#endif /* __CUVNET_SMART_PTR_HPP__ */
