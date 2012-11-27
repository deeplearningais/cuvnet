#ifndef __ARGSORT_HPP__
#     define __ARGSORT_HPP__
#include <functional>
#include <iterator>

namespace detail
{
    /**
     * comparator object for argsort
     */
    template<class T, class Cmp>
        struct argsort_cmp
        {
            T begin;
            Cmp cmp;
            argsort_cmp( T _begin, Cmp _cmp ) : begin( _begin), cmp(_cmp)  {}

            bool   operator() ( unsigned int a, unsigned int b  ) {
                return cmp(begin[a] , begin[b]);
            }
        };
}

/**
 * sort indices based on given comparator.
 *
 * @param begin start of range
 * @param end end of range
 * @param cmp comparator
 * @return sorted index list
 * @ingroup tools
 */
    template <typename RandomAccessIter, class Comparator>
    std::vector<unsigned int> argsort( RandomAccessIter begin, RandomAccessIter end, Comparator cmp)
    {
        detail::argsort_cmp<RandomAccessIter,Comparator> idx_cmp(begin, cmp); 
        std::vector<unsigned int> idx(std::distance(begin,end));
        unsigned int i=0;
        std::vector<unsigned int>::iterator it=idx.begin();
        while(it!=idx.end())
            *it++ = i++;
        std::sort(idx.begin(),idx.end(),idx_cmp);
        return idx;
    }
/**
 * sort indices based on values in the given range.
 *
 * @param begin start of range
 * @param end end of range
 * @return sorted index list
 * @ingroup tools
 */
    template <typename RandomAccessIter>
    inline
    std::vector<unsigned int> argsort( RandomAccessIter begin, RandomAccessIter end){
        return argsort(begin,end,std::less<typename std::iterator_traits<RandomAccessIter>::value_type>());
    }

#endif /* __ARGSORT_HPP__ */
