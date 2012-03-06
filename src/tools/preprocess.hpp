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
                data += m_add;
            }
            void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
    };

    class pca_whitening{
        private:
            int m_n_components;
            zero_mean_unit_variance<cuv::host_memory_space> m_zm;
            cuv::tensor<float,cuv::host_memory_space> m_rot;

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

        public:
            

            pca_whitening(int n_components=-1, bool whiten=true, bool zca=false):m_n_components(n_components),m_zm(false),m_whiten(whiten),m_zca(zca){}
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
                m_rot.resize(cuv::extents[train.shape(1)][m_n_components]);
                adaptor_matrix base_data(train.shape(0),train.shape(1),ArrayAdaptor(train.size(),const_cast<real*>(train.ptr())));
                adaptor_matrix ucov      (train.shape(1),train.shape(1),ArrayAdaptor(cov.size(),const_cast<real*>(cov.ptr())));
                adaptor_matrix rot      (train.shape(1),m_n_components,ArrayAdaptor(m_rot.size(),const_cast<real*>(m_rot.ptr())));

                vector S(train.shape(1));
                ubmatrix Eigv(ucov);
                vector lambda(train.shape(1));
                lapack::syev( 'V', 'L', Eigv, lambda, lapack::optimal_workspace() );
                std::vector<unsigned int> idx = argsort(lambda.begin(), lambda.end(), std::greater<float>());
                sort(lambda.begin(), lambda.end(), std::greater<float>());

                // calculate the cut-off Eigv and diagMatrix matrix 
                ublas::diagonal_matrix<real> diagMatrix(m_n_components);
                static const float epsilon = 0.00001f;
                if(m_whiten)
                    for(int i=0;i<m_n_components; i++){
                        real r = lambda(i) + epsilon;
                        diagMatrix(i,i) = 1.0/sqrt(r);
                    }
                else
                    for(int i=0;i<m_n_components; i++){
                        diagMatrix(i,i) = 1.f;
                    }
                for(int i=0;i<m_n_components;i++) {
                    boost::numeric::ublas::matrix_column<adaptor_matrix>(rot,i) = column(Eigv,idx[i]);
                }

                if(m_zca){
                    // ZCA whitening:
                    // Rot = rot * diagMatrix * rot'
                    ubmatrix tmp = ublas::prod(rot, diagMatrix);
                    rot = ublas::prod(tmp, ublas::trans(rot) );
                }
                else{
                    // PCA whitening:
                    // Rot = rot * diagMatrix
                    rot = ublas::prod(rot, diagMatrix);
                }
            }
            void transform(cuv::tensor<float,cuv::host_memory_space>& data){
                m_zm.transform(data);
                cuv::tensor<float,cuv::host_memory_space> res(cuv::extents[data.shape(0)][m_n_components]);
                cuv::prod( res, data, m_rot );
                data = res;
            }
            void fit_transform(cuv::tensor<float,cuv::host_memory_space>& data){
                fit(data); transform(data);
            }
            void reverse_transform(cuv::tensor<float,cuv::host_memory_space>& res){
                cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[res.shape(0)][m_rot.shape(0)]);
                cuv::prod( data, res, m_rot, 'n','t' );
                m_zm.reverse_transform(data);
                res = data;
            }

            inline
                const cuv::tensor<float,cuv::host_memory_space>& rot(){return m_rot;};
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
