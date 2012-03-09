#ifndef __CUVNET_PREPROCESS_HPP__
#     define __CUVNET_PREPROCESS_HPP__

#include<cuv/basics/tensor.hpp>
#include<cuv/matrix_ops/matrix_ops.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>

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

#include "argsort.hpp"
namespace cuvnet
{
    namespace detail
    {
        struct file_descr{
            std::string        name;
            size_t             size;
            std::vector<char> content;
        };
    }

    template<class M=cuv::host_memory_space>
    class zero_sample_mean{
        public:
        public:
            void fit(const cuv::tensor<float,M>& train){
            }
            void transform(cuv::tensor<float,M>& data){
                cuv::tensor<float,M> c(cuv::extents[data.shape(0)]);
                cuv::reduce_to_col(c,data,cuv::RF_ADD,0.f,-1.f); 
                c/=(float)data.shape(1);
                cuv::matrix_plus_col(data,c);
            }
            void reverse_transform(cuv::tensor<float,M>& data){
                throw std::runtime_error("zero_sample_mean cannot be inverted");
            }
            void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
    };

    template<class M=cuv::host_memory_space>
    class log_transformer{
        public:
        public:
            void fit(const cuv::tensor<float,M>& train){
            }
            void transform(cuv::tensor<float,M>& data){
                cuv::apply_scalar_functor(data,cuv::SF_LOG); 
            }
            void reverse_transform(cuv::tensor<float,M>& data){
                cuv::apply_scalar_functor(data,cuv::SF_EXP); 
            }
            void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
    };

    template<class M=cuv::host_memory_space>
    class zero_mean_unit_variance{
        public:
            cuv::tensor<float, M> m_mean;
            cuv::tensor<float, M> m_std;
            bool m_unitvar;
        public:
            zero_mean_unit_variance(bool unitvar=true):m_unitvar(unitvar){}
            void fit(const cuv::tensor<float,M>& train){
                using namespace cuv;
                m_mean.resize(extents[train.shape(1)]);
                if(m_unitvar)
                    m_std .resize(extents[train.shape(1)]);
                reduce_to_row(m_mean,train,RF_ADD);
                if(m_unitvar){
                    reduce_to_row(m_std, train,RF_ADD_SQUARED);
                    m_std  /= (float)train.shape(0);
                }
                m_mean /= (float)train.shape(0);
                if(m_unitvar){
                    m_std -= ::operator*(m_mean,m_mean);
                    apply_scalar_functor(m_std, SF_SQRT);
                    m_std += 0.01f; // regularized
                }
                apply_scalar_functor(m_mean, SF_NEGATE);
            }
            void transform(cuv::tensor<float,M>& data){
                cuv::matrix_plus_row(data,m_mean); // mean is negated already
                if(m_unitvar)
                    cuv::matrix_divide_row(data,m_std);
            }
            void reverse_transform(cuv::tensor<float,M>& data){
                using namespace cuv; // for operator-
                tensor<float,M> tmp(m_mean.shape());
                apply_scalar_functor(tmp,m_mean,SF_NEGATE);
                if(m_unitvar)
                    matrix_times_row(data,m_std);
                matrix_plus_row(data, tmp);
            }
            void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
    };

    template<class M=cuv::host_memory_space>
    class global_min_max_normalize{
        private:
            float m_min, m_max;
            float m_add, m_fact;
        public:
            global_min_max_normalize(float min=0.f, float max=1.f):m_min(min),m_max(max){}
            void fit(const cuv::tensor<float,M>& train){
                float xmin = cuv::minimum(train); // minimum
                float xmax = cuv::maximum(train); // range
                m_add  = (xmin*m_max + xmax*m_min)/(xmax-xmin);
                m_fact = (m_max-m_min)/(xmax-xmin);
                cuvAssert(xmax>xmin);
            }
            void transform(cuv::tensor<float,M>& data){
                data *= m_fact;
                data -= m_add;
            }
            void reverse_transform(cuv::tensor<float,M>& data){
                using namespace cuv; // for operator-
                data += m_add;
                data /= m_fact;
            }
            void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
    };

    class pca_whitening{
        private:
            int m_n_components;
            zero_mean_unit_variance <cuv::host_memory_space> m_zm;
            cuv::tensor<float,cuv::host_memory_space> m_rot_trans;
            cuv::tensor<float,cuv::host_memory_space> m_rot_revrs;

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

            bool m_whiten, m_zca;
            boost::numeric::ublas::diagonal_matrix<real> m_diag;
            float m_epsilon;

        public:
            

            /**
             * constructor.
             *
             * @param n_components   how many components to use for (zca: intermediate, else: resulting) representation
             * @param whiten         if true, divide by root of eigenvalue after rotation
             * @param zca            if true, rotate back into original space
             * @param epsilon        regularizer for whitening, about 0.01 or 0.1 if data is in range [0,1]
             */
            pca_whitening(int n_components=-1, bool whiten=true, bool zca=false, float epsilon=0.01)
                :m_n_components(n_components)
                 ,m_zm(false)
                 ,m_whiten(whiten)
                 ,m_zca(zca)
                 ,m_epsilon(epsilon)
                {}
            void fit(const cuv::tensor<float,cuv::host_memory_space>& train_){
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
                        m_diag(i,i) = 1.0/sqrt(lambda(idx[i])+m_epsilon);
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
            void transform(cuv::tensor<float,cuv::host_memory_space>& data){
                m_zm.transform(data);
                cuv::tensor<float,cuv::host_memory_space> res(cuv::extents[data.shape(0)][m_rot_trans.shape(1)]);
                cuv::prod( res, data, m_rot_trans );
                data = res;
            }
            void fit_transform(cuv::tensor<float,cuv::host_memory_space>& data){
                fit(data); transform(data);
            }
            void reverse_transform(cuv::tensor<float,cuv::host_memory_space>& res, bool nomean=false){
                cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[res.shape(0)][m_rot_revrs.shape(1)]);
                cuv::prod( data, res, m_rot_revrs);
                if(!nomean)
                    m_zm.reverse_transform(data);
                res = data;
            }

            inline
                const cuv::tensor<float,cuv::host_memory_space>& rot(){return m_rot_trans;};
            inline
                const cuv::tensor<float,cuv::host_memory_space>& rrot(){return m_rot_revrs;};
    };

    class preprocessor{
        public:
            virtual void process_filestring(cuv::tensor<float,cuv::host_memory_space>& dst, const char* buf, size_t n)=0;
    };

    class patch_extractor
    : public preprocessor
    {
        private:
            unsigned int m_s0, m_s1;
        public:
            patch_extractor(unsigned int s0, unsigned int s1):m_s0(s0),m_s1(s1){}
            void process_filestring(cuv::tensor<float,cuv::host_memory_space>& dst, const char* buf, size_t n);
    };
}
#endif /* __CUVNET_PREPROCESS_HPP__ */
