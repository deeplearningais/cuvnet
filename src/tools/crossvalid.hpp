#ifndef __CROSSVALID_HPP__
#     define __CROSSVALID_HPP__
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/limits.hpp>
#include <boost/bind.hpp>
#include <cuv/tools/device_tools.hpp>
#include "gradient_descent.hpp"

namespace cuvnet
{
    template<class T>
    struct crossvalidation;

    template<class T>
    struct split_runner
    {
        crossvalidation<T>* m_cv;
        
        void operator()(){
        }
    };
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

        boost::asio::io_service       m_io_service;
        boost::asio::io_service::work m_work;
        boost::thread_group           m_threads;

        /** 
         * do a for-loop over your params in here
         * creating a trainer for each and supply it
         * to evaluate
         */
        virtual void generate_and_test_models(){
        }

        /// triggered before calling fit/predict on trainer
        /// with parameters split and cv_mode
        boost::signal<void(T*,unsigned int, cv_mode)> switch_dataset;

        /// triggered when new best model is found
        boost::signal<void(T&, float)> new_best_model;

        /// triggered when new result is available
        boost::signal<void(T&, float)> new_result;

        boost::mutex m_best_model_mutex;
        void store_new_best(T& t, float v){
            boost::mutex::scoped_lock(m_best_model_mutex);
            if(v<m_best_model_perf) {
                std::cout <<"CV: new best model: "<< v<<std::endl;
                m_best_model      = t;
                m_best_model_perf = v;
            }
        }

        void evaluate(T& t){
            cuv::initCUDA(-1);
            std::cout << "working on device "<<cuv::getCurrentDevice()<<std::endl;
            cuv::tensor<float,cuv::dev_memory_space> tmp(cuv::extents[5][8]);
            tmp = 5.f;
            float sum = 0.f;
            for(unsigned int s=0;s<m_n_splits;s++){
                switch_dataset(&t,s,CM_TRAIN);
                t.fit();
                switch_dataset(&t,s,CM_VALID);
                sum += t.predict();
            }
            sum /= m_n_splits;
            new_result(t,sum);
        }

        float run(){
            generate_and_test_models();
            m_io_service.stop();
            m_threads.join_all();
            switch_dataset(&m_best_model, 0,CM_TEST);
            // TODO: create new model which has the same params as
            // m_best_model and train it on the complete training set
            std::cout << "warning: not trained on whole training set!"<<std::endl;
            float v = m_best_model.predict();
            return v;
        }
        
        /**
         * constructor
         *
         * @param n_splits the number of splits
         * @param n_workers number of worker threads
         */
        crossvalidation(unsigned int n_splits, unsigned int n_workers=3)
        : m_n_splits(n_splits)
        , m_best_model_perf(std::numeric_limits<float>::infinity())
        , m_work(m_io_service)
        {
            new_result.connect(boost::bind(&crossvalidation::store_new_best, this, _1, _2));
            for (unsigned int i = 0; i < n_workers; ++i)
                m_threads.create_thread(boost::bind(&boost::asio::io_service::run, &m_io_service));
        }

        /**
         * (virtual!) destructor
         */
        virtual ~crossvalidation(){}

    };
}
#endif /* __CROSSVALID_HPP__ */
