#include <iostream>
#include <fstream>

/// for PCA whitening
#define BOOST_UBLAS_SHALLOW_ARRAY_ADAPTOR  1
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/bindings/lapack/syev.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp> 

#include <tools/preprocess.hpp>


namespace cuvnet
{

void pca_whitening::fit(const cuv::tensor<float,cuv::host_memory_space>& train_){
    typedef boost::numeric::ublas::shallow_array_adaptor<real>  ArrayAdaptor; 
    typedef boost::numeric::ublas::matrix<real,boost::numeric::ublas::row_major,ArrayAdaptor>    adaptor_matrix; 
    typedef boost::numeric::ublas::matrix_column<adaptor_matrix>    acolumn; 
    typedef boost::numeric::ublas::matrix<real,boost::numeric::ublas::column_major> ubmatrix;
    typedef boost::numeric::ublas::matrix_column<ubmatrix>            column; 
    typedef boost::numeric::ublas::matrix_column<ubmatrix>            row; 
    typedef boost::numeric::ublas::vector<real>                     vector;
    typedef boost::numeric::ublas::scalar_vector<real>              scalar_vector;
    typedef boost::numeric::ublas::range                            range;

    boost::numeric::ublas::diagonal_matrix<real> m_diag;
    namespace lapack = boost::numeric::bindings::lapack;
    namespace ublas  = boost::numeric::ublas;
    if(m_n_components<=0)
        m_n_components = train_.shape(1);

    // ------- determine covariance matrix ----------
    //  subtract mean
    cuv::tensor<float,cuv::host_memory_space> train = train_.copy();
    m_zm.fit_transform(train);

    // calculate covariance matrix
    cuv::tensor<float,cuv::host_memory_space> cov(cuv::extents[train_.shape(1)][train_.shape(1)]);
    cuv::prod(cov,train,train,'t','n');
    cov /= (float)train.shape(0);

    // create UBLAS adaptors 
    unsigned int d2 = m_zca ? train.shape(1) : m_n_components;
    m_rot_trans.resize(cuv::extents[train.shape(1)][d2]);
    m_rot_revrs.resize(cuv::extents[d2][train.shape(1)]);
    adaptor_matrix ucov       (train.shape(1),train.shape(1),ArrayAdaptor(cov.size(),const_cast<real*>(cov.ptr())));
    adaptor_matrix rot_trans  (train.shape(1),d2,ArrayAdaptor(m_rot_trans.size(),const_cast<real*>(m_rot_trans.ptr())));
    adaptor_matrix rot_revrs  (d2,train.shape(1),ArrayAdaptor(m_rot_revrs.size(),const_cast<real*>(m_rot_revrs.ptr())));

    vector S(train.shape(1));
    ubmatrix Eigv(ucov);
    vector lambda(train.shape(1));
    lapack::syev( 'V', 'L', Eigv, lambda, lapack::optimal_workspace() );
    std::vector<unsigned int> idx = argsort(lambda.begin(), lambda.end(), std::greater<float>());

    // calculate the cut-off Eigv and m_diag matrix 
    m_diag = ublas::diagonal_matrix<real> (m_n_components);
    if(m_whiten)
        for(int i=0;i<m_n_components; i++)
            m_diag(i,i) = 1.0/std::sqrt(lambda(idx[i])+m_epsilon);
    else
        for(int i=0;i<m_n_components; i++)
            m_diag(i,i) = 1.f;

    ubmatrix rot(train.shape(1), m_n_components);
    for(int i=0;i<m_n_components;i++) 
        column(rot,i) = column(Eigv,idx[i]);

    std::cout <<"w:"<<m_whiten<<" z:"<<m_zca<<" n:"<<m_n_components<<" s:"<<train_.shape(1)<<std::endl;
    if(!m_whiten && m_zca && (unsigned int)m_n_components == train_.shape(1)){
        // don't do anything!
        ublas::noalias(rot_trans) = ublas::identity_matrix<float>(m_n_components);
        ublas::noalias(rot_revrs) = ublas::identity_matrix<float>(m_n_components);
    }
    else if(m_zca){
        // ZCA whitening:
        // Rot = rot * m_diag * rot'
        ubmatrix tmp = ublas::prod(rot, m_diag);
        ublas::noalias(rot_trans) = ublas::prod(tmp, ublas::trans(rot) );

        // invert
        for (int i = 0; i < m_n_components; ++i)
            m_diag(i,i) = 1.0/m_diag(i,i);
        tmp = ublas::prod(rot,m_diag);
        ublas::noalias(rot_revrs) = ublas::prod(tmp,ublas::trans(rot));
    }
    else{
        // PCA whitening:
        // Rot = rot * m_diag
        ublas::noalias(rot_trans) = ublas::prod(rot, m_diag);

        // invert
        for (int i = 0; i < m_n_components; ++i)
            m_diag(i,i) = 1.0/m_diag(i,i);
        ublas::noalias(rot_revrs) = ublas::prod(m_diag,ublas::trans(rot));
    }
}

}
