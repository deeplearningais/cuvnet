
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


template<class T>
void orthogonalize_symmetric_(cuv::tensor<float, T>& m, bool columns){
    namespace ublas = boost::numeric::ublas; 
    namespace lapack = boost::numeric::bindings::lapack;
    cuvAssert(m.ndim()==2);

    unsigned int n = columns ? m.shape(0) : m.shape(1);

    std::cout << "columns:" << columns << " n:" << n << std::endl;

    // determine correlations between columns/rows
    cuv::tensor<float,T> res(cuv::extents[n][n]);
    if(columns) cuv::prod(res, m, m, 'n', 't');
    else        cuv::prod(res, m, m, 't', 'n');

    // ensure we continue on host
    cuv::tensor<float,cuv::host_memory_space> hres = res;
    res.dealloc();
    adaptor_matrix hresa       (hres.shape(0),hres.shape(1),ArrayAdaptor(hres.size(),const_cast<float*>(hres.ptr())));
    ubmatrix A(hresa);

    unsigned int minmn = std::min(A.size1(), A.size2());
    ubmatrix U (A.size1(), minmn);  
    vector   S(minmn);
    ubmatrix Vt(minmn,A.size2());

    lapack::gesdd('S',A,S,U,Vt);

    ublas::diagonal_matrix<real> diagMatrix(S.size());
    for(unsigned int i=0;i<S.size(); i++){
        if(S(i)>0.000001) diagMatrix(i,i) = 1.0/sqrt(S(i));
        else              diagMatrix(i,i) = 0.;
    }
    ubmatrix v2 = ublas::prod(diagMatrix, ublas::trans(U));
    ublas::noalias(hresa) = ublas::prod(ublas::trans(Vt),v2);

    // copy back to device
    cuv::tensor<float,T> invres = hres;
    std::cout << "m.shape(0):" << m.shape(0) << " m.shape(1):" << m.shape(1) << " invres.shape(0):" << invres.shape(0) << " invres.shape(1):" <<  invres.shape(1)<< std::endl;
    if(columns) cuv::prod(m,invres,m.copy());
    else        cuv::prod(m,m.copy(),invres,'n','t');
}
void orthogonalize_symmetric(cuv::tensor<float, cuv::host_memory_space>& m, bool columns){
    orthogonalize_symmetric_(m,columns);
}
void orthogonalize_symmetric(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns){
    orthogonalize_symmetric_(m,columns);
}
void orthogonalize_symmetric(cuv::tensor<float, cuv::host_memory_space>& m){
    namespace ublas = boost::numeric::ublas; 
    namespace lapack = boost::numeric::bindings::lapack;
    cuvAssert(m.ndim()==2);

    adaptor_matrix m_       (m.shape(0),m.shape(1),ArrayAdaptor(m.size(),const_cast<float*>(m.ptr())));
    ubmatrix A(m_);

    unsigned int minmn = std::min(A.size1(), A.size2());
    ubmatrix U (A.size1(), minmn);  
    vector   S(minmn);
    ubmatrix Vt(minmn,A.size2());

    lapack::gesdd('S',A,S,U,Vt);

    ublas::noalias(m_) = ublas::prod(U,Vt);
}

void orthogonalize_lowdin(cuv::tensor<float, cuv::host_memory_space>& m){
    namespace ublas = boost::numeric::ublas; 
    namespace lapack = boost::numeric::bindings::lapack;
    cuvAssert(m.ndim()==2);

    adaptor_matrix m_       (m.shape(0),m.shape(1),ArrayAdaptor(m.size(),const_cast<float*>(m.ptr())));
    ubmatrix A(m_);

    unsigned int minmn = std::min(A.size1(), A.size2());
    ubmatrix U (A.size1(), minmn);  
    vector   S(minmn);
    ubmatrix Vt(minmn,A.size2());

    lapack::gesdd('S',A,S,U,Vt);

    ublas::noalias(m_) = ublas::prod(U,Vt);
}

template<class T>
void orthogonalize_pairs_(cuv::tensor<float, T>& m, bool columns){
    if(columns){
        for (unsigned int i = 0; i < m.shape(1)-1; i += 2) {
            cuv::tensor<float,cuv::host_memory_space> pair
                = m[cuv::indices[cuv::index_range()][cuv::index_range(i,i+2)]].copy();
            orthogonalize_symmetric_(pair,true);
            m[cuv::indices[cuv::index_range()][cuv::index_range(i,i+2)]] = pair;
        }
    }else{
        for (unsigned int i = 0; i < m.shape(0)-1; i += 2) {
            cuv::tensor<float,cuv::host_memory_space> pair
                = m[cuv::indices[cuv::index_range(i,i+2)][cuv::index_range()]].copy();
            orthogonalize_symmetric_(pair,true);
            m[cuv::indices[cuv::index_range(i,i+2)][cuv::index_range()]] = pair;
        }
    }
}

void orthogonalize_pairs(cuv::tensor<float, cuv::host_memory_space>& m, bool columns){
    orthogonalize_pairs_(m,columns);
}
void orthogonalize_pairs(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns){
    orthogonalize_pairs_(m,columns);
}

void orthogonalize(cuv::tensor<float, cuv::dev_memory_space>& m, bool columns){
    if(columns) m = trans___o16n(m);

    cuv::tensor<float,cuv::host_memory_space> mh = m;
    orthogonalize_lowdin(mh);
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
