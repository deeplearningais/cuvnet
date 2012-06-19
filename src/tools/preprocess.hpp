#ifndef __CUVNET_PREPROCESS_HPP__
#     define __CUVNET_PREPROCESS_HPP__

#include<boost/foreach.hpp>

#include<cuv/basics/tensor.hpp>
#include<cuv/matrix_ops/matrix_ops.hpp>
#include<cuv/tensor_ops/tensor_ops.hpp>

#include "matwrite.hpp"

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

    /**
     * Base class for pre-processing.
     * @ingroup preproc
     */
    template<class M=cuv::host_memory_space>
    class preprocessor{
        public:
            /// should determine transformation parameters, but not change the parameters
            virtual void fit(const cuv::tensor<float,M>& train)=0;
            /// transform the data using the parameters determined in \c fit()
            virtual void transform(cuv::tensor<float,M>& data)=0;
            /// reverse transform (often this is not possible!)
            virtual void reverse_transform(cuv::tensor<float,M>& data)=0;
            /// fit and transform (sometimes this is not just shorter, but also more efficient!)
            virtual void fit_transform(cuv::tensor<float,M>& data){
                fit(data); transform(data);
            }
            /// write the transformation parameters to a file
            virtual void write_params(const std::string& basefn){}
    };

    /**
     * Convenience class for performing multiple consecutive pre-processing steps.
     * @ingroup preproc
     */
    template<class M=cuv::host_memory_space>
    class preprocessing_pipeline{
        protected:
            std::vector<boost::shared_ptr<preprocessor<M> > > m_pp;
        public:

            /// add an element to the pipe
            preprocessing_pipeline<M>& add(preprocessor<M>* pp){
                m_pp.push_back(boost::shared_ptr<preprocessor<M> >(pp));
                return *this;
            }
            /**
             * fit using all elements in the pipe.
             *
             * @note If you want to transform the training data as well, it is more
             * efficient to call \c fit_transform() directly.
             *
             * @param train the data to be fitted
             */
            virtual void fit(const cuv::tensor<float, M>& train){
                cuv::tensor<float, M> cpy = train.copy();
                for(auto it=m_pp.begin(); it!=m_pp.end(); it++){
                    preprocessor<M>& pp = **it;
                    pp.fit(cpy);
                    if(it != m_pp.end()-1) // do not transform last instance
                        pp.transform(cpy);
                }
            }
            /**
             * fit and transform using all elements in the pipe.
             *
             * @note If you want to transform the training data as well, it is more
             * efficient to call \c fit_transform() directly.
             *
             * @param train the data to be fitted
             */
            virtual void fit_transform(cuv::tensor<float, M>& train){
                for(auto it=m_pp.begin(); it!=m_pp.end(); it++){
                    preprocessor<M>& pp = **it;
                    pp.fit(train);
                    pp.transform(train);
                }
            }
            /**
             * transform according to the fitted parameters.
             * @param val the data to be transformed
             */
            virtual void transform(cuv::tensor<float, M>& val){
                for(auto it=m_pp.begin(); it!=m_pp.end(); it++){
                    preprocessor<M>& pp = **it;
                    pp.transform(val);
                }
            }
    };

    /**
     * Does nothing.
     * @ingroup preproc
     */
    template<class M=cuv::host_memory_space>
    class identity_preprocessor
    : public preprocessor<M>
    {
        public:
            void fit(const cuv::tensor<float,M>& train){};
            void transform(cuv::tensor<float,M>& data){};
            void reverse_transform(cuv::tensor<float,M>& data){};
    };

    /**
     * Ensures that every sample (=row) has mean zero.
     *
     * This cannot be reversed, since the means are either not remembered or not known.
     * @ingroup preproc
     */
    template<class M=cuv::host_memory_space>
    class zero_sample_mean : public preprocessor<M> {
        public:
        public:
            /// @overload
            void fit(const cuv::tensor<float,M>& train){
            }
            /// @overload
            void transform(cuv::tensor<float,M>& data){
                cuv::tensor<float,M> c(cuv::extents[data.shape(0)]);
                cuv::reduce_to_col(c,data,cuv::RF_ADD,-1.f,0.f); 
                c/=(float)data.shape(1);
                cuv::matrix_plus_col(data,c);
            }
            /// @overload
            void reverse_transform(cuv::tensor<float,M>& data){
                throw std::runtime_error("zero_sample_mean cannot be inverted");
            }
    };

    /**
     * Takes the logarithm of all values in the dataset.
     * @ingroup preproc
     */
    template<class M=cuv::host_memory_space>
    class log_transformer : public preprocessor<M> {
        public:
            /// @overload
            void fit(const cuv::tensor<float,M>& train){
            }
            /// @overload
            void transform(cuv::tensor<float,M>& data){
                cuv::apply_scalar_functor(data,cuv::SF_LOG); 
            }
            /// @overload
            void reverse_transform(cuv::tensor<float,M>& data){
                cuv::apply_scalar_functor(data,cuv::SF_EXP); 
            }
    };

    /**
     * Ensures that each column (=variable) has zero mean and unit variance.
     * @ingroup preproc
     */
    template<class M=cuv::host_memory_space>
    class zero_mean_unit_variance : public preprocessor<M>{
        public:
            cuv::tensor<float, M> m_mean;
            cuv::tensor<float, M> m_std;
            bool m_unitvar;
        public:
            /// @overload
            zero_mean_unit_variance(bool unitvar=true):m_unitvar(unitvar){}
            /// @overload
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
            /// @overload
            void transform(cuv::tensor<float,M>& data){
                cuv::matrix_plus_row(data,m_mean); // mean is negated already
                if(m_unitvar)
                    cuv::matrix_divide_row(data,m_std);
            }
            /// @overload
            void reverse_transform(cuv::tensor<float,M>& data){
                using namespace cuv; // for operator-
                tensor<float,M> tmp(m_mean.shape());
                apply_scalar_functor(tmp,m_mean,SF_NEGATE);
                if(m_unitvar)
                    matrix_times_row(data,m_std);
                matrix_plus_row(data, tmp);
            }
    };

    /**
     * Transforms the data such that it is between given minimum and maximum values
     * @ingroup preproc
     */
    template<class M=cuv::host_memory_space>
    class global_min_max_normalize : public preprocessor<M> {
        private:
            float m_min, m_max;
            float m_add, m_fact;
        public:
            /// @overload
            global_min_max_normalize(float min=0.f, float max=1.f):m_min(min),m_max(max){}
            /// @overload
            void fit(const cuv::tensor<float,M>& train){
                float xmin = cuv::minimum(train); // minimum
                float xmax = cuv::maximum(train); // range
                m_add  = (xmin*m_max + xmax*m_min)/(xmax-xmin);
                m_fact = (m_max-m_min)/(xmax-xmin);
                cuvAssert(xmax>xmin);
            }
            /// @overload
            void transform(cuv::tensor<float,M>& data){
                data *= m_fact;
                data -= m_add;
            }
            /// @overload
            void reverse_transform(cuv::tensor<float,M>& data){
                using namespace cuv; // for operator-
                data += m_add;
                data /= m_fact;
            }
    };

    /**
     * PCA transformation, dimensionality reduction and ZCA whitening
     * @ingroup preproc
     */
    class pca_whitening : public preprocessor<cuv::host_memory_space> {
        private:
            int m_n_components;
            zero_mean_unit_variance <cuv::host_memory_space> m_zm;
            cuv::tensor<float,cuv::host_memory_space> m_rot_trans;
            cuv::tensor<float,cuv::host_memory_space> m_rot_revrs;

            // shorthands
            typedef float real;

            bool m_whiten, m_zca;
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
            /// @overload
            void fit(const cuv::tensor<float,cuv::host_memory_space>& train_);

            /// @overload
            void transform(cuv::tensor<float,cuv::host_memory_space>& data){
                m_zm.transform(data);
                cuv::tensor<float,cuv::host_memory_space> res(cuv::extents[data.shape(0)][m_rot_trans.shape(1)]);
                cuv::prod( res, data, m_rot_trans );
                data = res;
            }
            /// @overload
            void reverse_transform(cuv::tensor<float,cuv::host_memory_space>& res){
                reverse_transform(res, false);
            }
            /// @overload
            void reverse_transform(cuv::tensor<float,cuv::host_memory_space>& res, bool nomean){
                cuv::tensor<float,cuv::host_memory_space> data(cuv::extents[res.shape(0)][m_rot_revrs.shape(1)]);
                cuv::prod( data, res, m_rot_revrs);
                if(!nomean)
                    m_zm.reverse_transform(data);
                res = data;
            }
            /// @overload
            void write_params(const std::string& basefn){
                tofile(basefn+"-mean.npy", m_zm.m_mean);
                tofile(basefn+"-rot-trans.npy", m_rot_trans);
                tofile(basefn+"-rot-revrs.npy", m_rot_revrs);
            }

            inline
                const cuv::tensor<float,cuv::host_memory_space>& rot(){return m_rot_trans;};
            inline
                const cuv::tensor<float,cuv::host_memory_space>& rrot(){return m_rot_revrs;};
    };

    class patch_extractor
    //: public preprocessor
    {
        private:
            unsigned int m_s0, m_s1;
        public:
            patch_extractor(unsigned int s0, unsigned int s1):m_s0(s0),m_s1(s1){}
            void process_filestring(cuv::tensor<float,cuv::host_memory_space>& dst, const char* buf, size_t n);
    };
}
#endif /* __CUVNET_PREPROCESS_HPP__ */
