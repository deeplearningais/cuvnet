// vim:ts=4:sw=4:et
#ifndef __CUVNET_SMART_PTR_HPP__
#     define __CUVNET_SMART_PTR_HPP__
#include <ostream>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/version.hpp>
#include <boost/make_shared.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <cuv/basics/tensor.hpp>

namespace cuvnet
{
    namespace detail {
        template<class T>
            struct cow_ptr_traits{
                typedef void unload_t;
                static T* clone(const T& t){ return new T(t); }
                static unload_t* unload_from_dev(T& t){
                    throw std::runtime_error("unload from dev not implemented for this type");
                }
                static void check(const T*){}
                template<class Archive>
                static void serialize_host(Archive& ar, boost::shared_ptr<void>&){}
            };
        template<class V, class M, class L>
            struct cow_ptr_traits<cuv::tensor<V,M,L> >{
                typedef cuv::tensor<V,M,L> T;
                typedef cuv::tensor<V,cuv::host_memory_space,L> OT;
                typedef OT unload_t;
                static T* clone(const T& t); // { return new T(t,cuv::linear_memory_tag()); }
                static void check(const T* t){
                    if(t){
                        cuvAssert(!cuv::has_nan(*t));
                        cuvAssert(!cuv::has_inf(*t));
                    }
                }
                static unload_t* unload_from_dev(boost::shared_ptr<cuv::tensor<V,cuv::host_memory_space,L> >& t){
                    return NULL;
                }
                static unload_t* unload_from_dev(boost::shared_ptr<cuv::tensor<V,cuv::dev_memory_space,L> >& t){
                    unload_t* ot = new unload_t(*t);
                    t.reset();
                    return ot;
                }
                template<class Archive, class OT, class OL>
                static void serialize_host(Archive& ar, boost::shared_ptr<void>&){}

                template<class Archive, class OT, class OL>
                static void serialize_host(Archive& ar, boost::shared_ptr<cuv::tensor<OT, cuv::host_memory_space, OL> >& ot){
                    ar & ot;
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
     *
     * @ingroup tools
     */
    template <class T>
        class cow_ptr
        {
            public:
                /// type of the contained data -- a shared pointer!
                typedef boost::shared_ptr<T> ref_ptr;
                static boost::shared_ptr<cuv::allocator> s_allocator;

                /// @name constructors
                /// @{

                /// constructor (does nothing)
                cow_ptr( ){}
                /// copy constructor (cheap)
                cow_ptr( const cow_ptr &cpy ) : m_ptr(cpy.m_ptr){}
                /// create from shared_ptr (cheap)
                explicit cow_ptr( const ref_ptr &cpy ) : m_ptr(cpy){}
                /// create from raw pointer (cheap, takes ownership)
                explicit cow_ptr(       T*       cpy ) : m_ptr(cpy){}

                /// @}

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
                        m_ptr.reset(new T(tmp->shape(), s_allocator));
                    }
                }

                /// @name copying
                /// @{

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
                /// @}

                /// return internal pointer. Can be used to get around COW restrictions, use w/ care!
                T* ptr(){ return  m_ptr.get(); }

                /// @name data access: const versions
                /// @{
                const T&       data() const { return *m_ptr; } ///< const contained data, never detaches
                const T&      cdata() const { return *m_ptr; } ///< const contained data, never detaches
                const T& operator* () const { return *m_ptr; } ///< const contained data, never detaches
                const T* operator->() const { return  m_ptr.operator->(); } ///< const contained data, never detaches
                operator const T& ()const{ return *m_ptr; } ///< conversion to const contained data, never detaches
                bool     operator!()const   { return !m_ptr; } ///< true if pointing to NULL
                /// @}

                /// \name data access: non-const versions
                /// @{
                T&        data_onlyshape()  { detach_onlyshape(); return *m_ptr; } ///< contained data, detaches if needed, but retains only the /shape/ of the input
                T&        data()         { detach(); return *m_ptr; } ///< contained data, detaches if needed
                const T& cdata()         {           return *m_ptr; } ///< contained data, detaches if needed
                T& operator* ()          { detach(); return *m_ptr; } ///< contained data, detaches if needed
                T* operator->()          { detach(); return  m_ptr.operator->(); } ///< contained data, detaches if needed
                operator T& ()  { detach(); return *m_ptr; } ///< conversion to contained data, detaches if needed
                /// @}

                /** \name saving memory on GPU: unloading to host (costs time!)
                 *
                 * to avoid a constant overhead, you have to do this /explicitly/.
                 * Also, it only works if there is just one referrer to avoid
                 * arbitrary numbers of copies.
                 * @{
                 */
                void unload_from_dev(){
                    if(!unique())
                        return;
                    m_host_ptr.reset(detail::cow_ptr_traits<T>::unload_from_dev(m_ptr));
                }
                void ensure_on_dev(){
                    if(m_ptr)
                        return;
                    assert(m_host_ptr);
                    m_ptr.reset(new T(*m_host_ptr));
                }
                /**
                 * @}
                 */

                template<class U>
                friend std::ostream& operator<< (std::ostream &o, const cow_ptr<U>&);

            private:
                ref_ptr m_ptr;
                typedef typename detail::cow_ptr_traits<T>::unload_t host_ptr_value_type;
                typedef boost::shared_ptr<host_ptr_value_type> host_ptr;
                host_ptr m_host_ptr;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & m_ptr;
                        if(version > 0)
                            detail::cow_ptr_traits<T>::serialize_host(ar, m_host_ptr);
                    }
        };

    boost::shared_ptr<cuv::allocator>
        get_global_allocator();

    template<class T>
        boost::shared_ptr<cuv::allocator>
        cow_ptr<T>::s_allocator(get_global_allocator());

        template<class T>
        std::ostream&
        operator<< (std::ostream &o, const cow_ptr<T>& p){
            o << (T)p; return o;
        }

        namespace detail
        {
            template<class V, class M, class L>
                cuv::tensor<V,M,L>* cow_ptr_traits<cuv::tensor<V,M,L> >::
                clone(const cuv::tensor<V,M,L> & t)
                {
                    // the linear_memory_tag argument forces a copy of memory
                    return new cuv::tensor<V,M,L>(t,  cuv::linear_memory_tag());
                }
        }

        template<class T, class Arg>
        cow_ptr<T> make_cow_ptr(const Arg& a){
            return cow_ptr<T>(new T(a));
        }
}
namespace boost {
namespace serialization {
template<class T>
struct version< cuvnet::cow_ptr<T> >
{
    typedef mpl::int_<1> type;
    typedef mpl::integral_c_tag tag;
    BOOST_STATIC_CONSTANT(int, value = version::type::value);
    BOOST_MPL_ASSERT((
        boost::mpl::less<
            boost::mpl::int_<1>,
            boost::mpl::int_<256>
        >
    ));
};
}
}

#endif /* __CUVNET_SMART_PTR_HPP__ */
