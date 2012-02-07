#ifndef __CROSSVALID_HPP__
#     define __CROSSVALID_HPP__
#include <boost/asio.hpp>
//#include <boost/asio/signal_set.hpp>
#include <boost/thread.hpp>
#include <boost/limits.hpp>
#include <boost/bind.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include <cuv/tools/device_tools.hpp>
#include <cuv/tools/cuv_general.hpp>
#include "gradient_descent.hpp"
#include <cuda.h>

namespace cuvnet
{
    template<class T>
    struct crossvalidation;

    struct cuda_thread
    {
        boost::asio::io_service& m_io_service;
        int m_dev;
        cuda_thread(boost::asio::io_service& s, int dev):m_io_service(s),m_dev(dev){ }
        void operator()(){ 
            cuvSafeCall(cudaThreadExit());
            std::cout << "switching to device "<<m_dev<<std::endl;
            cuv::initCUDA(m_dev);
            cuv::initialize_mersenne_twister_seeds(time(NULL)+m_dev);
            m_io_service.run(); 
        }
    };

    template<class T>
    struct crossvalidation{

        enum cv_mode {
            CM_TRAIN,
            CM_TRAINALL,
            CM_VALID,
            CM_TEST
        };
        
        unsigned int m_n_splits;
        unsigned int m_n_workers;
        std::string  m_best_model;
        float m_best_model_perf;

        boost::asio::io_service       m_io_service;
        boost::thread_group           m_threads;
        //boost::asio::signal_set       m_signals;

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
        boost::signal<void(const std::string&, float)> new_best_model;

        /// triggered when new result is available
        boost::signal<void(const std::string&, float)> new_result;

        boost::mutex m_best_model_mutex;
        void store_new_best(const std::string& s, float v){
            boost::mutex::scoped_lock(m_best_model_mutex);
            if(v<m_best_model_perf) {
                //std::cout <<"CV: new best model: "<< v<<std::endl;
                m_best_model      = s;
                m_best_model_perf = v;
            }
        }
        void tostr(std::string& s, const T& t){
            std::ostringstream ss;
            boost::archive::binary_oarchive oa(ss);
            register_objects(oa);
            oa << t;
            s = ss.str();
        }
        void fromstr(T& t, const std::string& s){
            std::istringstream ss(s);
            boost::archive::binary_iarchive ia(ss);
            register_objects(ia);
            ia >> t;
        }

        void evaluate(const std::string& tstr){
            T t;
            fromstr(t,tstr);
            float sum = 0.f;
            for(unsigned int s=0;s<m_n_splits;s++){
                switch_dataset(&t,s,CM_TRAIN);
                t.reset_params();
                t.fit();
                switch_dataset(&t,s,CM_VALID);
                sum += t.predict();
            }
            sum /= m_n_splits;
            std::string s;
            tostr(s,t);
            std::cout <<"CV: result: "<< sum << " -- " << t.desc()<<std::endl;
            if(sum<m_best_model_perf)
                std::cout <<"CV: new best model: "<< sum << " -- " << t.desc()<<std::endl;
            new_result(s,sum);
        }

        void dispatch(T& t){
            std::string s;
            tostr(s,t);
            //m_best_model = s; std::cout <<t.desc()<<std::endl; return;
            if(m_n_workers>1)
                m_io_service.post(boost::bind(&crossvalidation<T>::evaluate, this, s));
            else
                evaluate(s);
        }

        float run(){
            {
                boost::asio::io_service::work work(m_io_service);
                generate_and_test_models();
            }
            m_io_service.run();
            m_io_service.stop();
            m_threads.join_all();
            T t;
            fromstr(t, m_best_model);

            switch_dataset(&t, 0,CM_TRAINALL);
            t.reset_params();
            t.fit();

            switch_dataset(&t, 0,CM_TEST);
            float v = t.predict();
            std::cout << "best result: "<< v<<": "<<t.desc()<<std::endl;
            return v;
        }
        
        /**
         * constructor
         *
         * @param n_splits the number of splits
         * @param n_workers number of worker threads
         */
        crossvalidation(unsigned int n_splits, unsigned int n_workers=1, unsigned int startdev=0)
        : m_n_splits(n_splits)
        , m_n_workers(n_workers)
        , m_best_model_perf(std::numeric_limits<float>::infinity())
        //, m_signals(m_io_service, SIGINT, SIGTERM)
        {
            /*
             *m_signals.async_wait(
             *        boost::bind(&boost::asio::io_service::stop, &m_io_service));
             *m_signals.async_wait(
             *        boost::bind(&boost::asio::io_service::reset, &m_io_service));
             */
            new_result.connect(boost::bind(&crossvalidation::store_new_best, this, _1, _2));
            if(n_workers>1)
                for (unsigned int i = startdev; i < n_workers; ++i) // dev 0 taken by main
                    //m_threads.create_thread(boost::bind(&boost::asio::io_service::run, &m_io_service));
                    m_threads.create_thread(cuda_thread(m_io_service, i));
        }

        /**
         * (virtual!) destructor
         */
        virtual ~crossvalidation(){}

    };
}
#endif /* __CROSSVALID_HPP__ */
