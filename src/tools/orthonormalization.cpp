
#define BOOST_UBLAS_SHALLOW_ARRAY_ADAPTOR  1
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/lapack/gesdd.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp> 

#include "orthonormalization.hpp"
#include <magma.h>

// shorthands
typedef float real;

typedef boost::numeric::ublas::shallow_array_adaptor<real>  ArrayAdaptor; 
typedef boost::numeric::ublas::matrix<real,boost::numeric::ublas::row_major,ArrayAdaptor>    adaptor_matrix; 
typedef boost::numeric::ublas::matrix_column<adaptor_matrix>    acolumn; 
typedef boost::numeric::ublas::matrix<real,boost::numeric::ublas::column_major> ubmatrix;
typedef boost::numeric::ublas::matrix_column<ubmatrix>            column; 
typedef boost::numeric::ublas::matrix_column<ubmatrix>            row; 
typedef boost::numeric::ublas::vector<real>                     vector;
typedef boost::numeric::ublas::scalar_vector<real>              scalar_vector;
typedef boost::numeric::ublas::range                            range;

template<class T, class M>
cuv::tensor<T,M> trans___o16n(cuv::tensor<T,M>& m){
    cuv::tensor<T,M> mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}


void orthogonalize_lowdin(cuv::tensor<float, cuv::host_memory_space>& m){
    namespace ublas = boost::numeric::ublas; 
    namespace lapack = boost::numeric::bindings::lapack;
    cuvAssert(m.ndim()==2);

    adaptor_matrix m_       (m.shape(0),m.shape(1),ArrayAdaptor(m.size(),const_cast<float*>(m.ptr())));
    ubmatrix A(m_);

    ubmatrix U (A.size1(),A.size2());  
    vector   S(std::min(A.size1(), A.size2()));
    ubmatrix Vt(A.size2(),A.size2());

    lapack::gesdd('S',A,S,U,Vt);
    ublas::noalias(m_) = ublas::prod(U,Vt);
}

void orthogonalize_pairs(cuv::tensor<float, cuv::host_memory_space>& m){
    for (unsigned int i = 0; i < m.shape(1)-1; i+=2) {
        cuv::tensor<float,cuv::host_memory_space> pair
            = m[cuv::indices[cuv::index_range()][cuv::index_range(i,i+2)]].copy();
        orthogonalize_lowdin(pair);
        m[cuv::indices[cuv::index_range()][cuv::index_range(i,i+2)]] = pair;
    }
}

void orthogonalize(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns){
    if(columns) m = trans___o16n(m);

    cuv::tensor<float,cuv::host_memory_space> mh = m;
    orthogonalize_lowdin(mh);
    m = mh;

    if(columns) m = trans___o16n(m);
}

void orthogonalize_pairs(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns){
    if(columns) m = trans___o16n(m);

    cuv::tensor<float,cuv::host_memory_space> mh = m;
    orthogonalize_pairs(mh);
    m = mh;

    if(columns) m = trans___o16n(m);
}

void orthonormalize_gramschmidt(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns){
    cuvAssert(m.ndim()==2);
    if(columns)
        m = trans___o16n(m);

/*
 * http://icl.cs.utk.edu/magma/docs/sgeqrf__gpu_8cpp.html#a5e82d8cd7bf269cbf4da7166bd672c64
 *
 *magma_int_t magma_sgeqrf_gpu    (   
 *        magma_int_t     m,
 *        magma_int_t     n,
 *        float *     dA,
 *        magma_int_t     ldda,
 *        float *     tau,
 *        float *     dT,
 *        magma_int_t *   info 
 *    ) 
 */


// m is row-major matrix here, but sgeqrf needs column-major
    magma_int_t info;
    cuv::tensor<float,cuv::host_memory_space> tau(std::min(m.shape(0),m.shape(1)));
    int NB = magma_get_sgeqrf_nb(m.shape(1));
    cuv::tensor<float,cuv::dev_memory_space> work(tau.shape(0)*2 + (m.shape(0)+31)/32*32*NB);
    magma_sgeqrf_gpu(m.shape(1), m.shape(0), m.ptr(), m.shape(1), tau.ptr(),work.ptr(), &info);
    cuvAssert(info==0);

/*
 *magma_int_t magma_sorgqr_gpu    (   
 *        magma_int_t     m,
 *        magma_int_t     n,
 *        magma_int_t     k,
 *        float *     da,
 *        magma_int_t     ldda,
 *        float *     tau,
 *        float *     dT,
 *        magma_int_t     nb,
 *        magma_int_t *   info 
 *    )       
 */
    magma_sorgqr_gpu(m.shape(1),m.shape(0),m.shape(0), m.ptr(), m.shape(1), tau.ptr(), work.ptr(),NB,&info);
    cuvAssert(info==0);

    if(columns)
        m = trans___o16n(m);
}
