#ifndef __CROSSVALID_HPP__
#     define __CROSSVALID_HPP__
#include "gradient_descent.hpp"
#include <boost/bind.hpp>

namespace cuvnet
{
    template<class T>
    struct crossvalidation{
        enum cv_mode {
            CM_TRAIN,
            CM_VALID,
            CM_TEST
        };
        
        unsigned int m_n_splits;
        T     m_best_model;
        float m_best_model_perf;

        virtual void generate_and_test_models(){
            // do a for-loop over your params in here
            // creating a trainer for each and supply it
            // to evaluate
        }

        /// triggered before calling fit/predict on trainer
        /// with parameters split and cv_mode
        boost::signal<void(unsigned int, cv_mode)> switch_dataset;

        /// triggered when new best model is found
        boost::signal<void(T&, float)> new_best_model;

        /// triggered when new result is available
        boost::signal<void(T&, float)> new_result;

        void store_new_best(T& t, float v){
            m_best_model      = t;
            m_best_model_perf = v;
        }

        void evaluate(T& t){
            for(unsigned int s=0;s<m_n_splits;i++){
                switch_dataset(s,CM_TRAIN);
                t.fit();
                switch_dataset(s,CM_VALID);
                float v = t.predict();
                new_result(t,v);
                if(v<m_best_model_perf) 
                    new_best_model(t,v);
            }
        }

        float run(){
            generate_and_test_models();
            switch_dataset(s,CM_TEST);
            // TODO: create new model which has the same params as
            // m_best_model and train it on the complete training set
            std::cout << "warning: not trained on whole training set!"<<std::endl;
            float v = m_best_model.predict();
            return v;
        }
        
        /**
         * constructor
         */
        crossvalidation(unsigned int n_splits)
        : m_n_splits(n_splits)
        , m_best_model(NULL)
        , m_best_model_perf(FLT_MAX);
        {
            new_best_model.connect(boost::bind(&crossvalidation::store_new_best, this));
        }
        virtual ~crossvalidation(){}

    };
}
#endif /* __CROSSVALID_HPP__ */
