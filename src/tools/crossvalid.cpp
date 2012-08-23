#include <stdexcept>
#include <boost/thread.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include <mongo/bson/bson.h>

#include<cuda.h>

#include<cuv.hpp>
#include<cuv/tools/device_tools.hpp>
#include<cuv/tools/cuv_general.hpp>

#include "crossvalid.hpp"

namespace cuvnet
{
	cv::crossvalidation_worker* g_worker;

	void cuda_thread::operator()(){ 
		cuvSafeCall(cudaThreadExit());
		std::cout << "switching to device "<<m_dev<<std::endl;
		cuv::initCUDA(m_dev);
		cuv::initialize_mersenne_twister_seeds(time(NULL)+m_dev);
		m_io_service.run(); 
	}

    bool crossvalidatable::refit_for_test()const{
        return true;
    }
    float crossvalidatable::refit_thresh()const{
        return INT_MAX;
    }

	namespace cv{
		one_split_evaluator::one_split_evaluator(unsigned int split, boost::shared_ptr<crossvalidatable> p){
			m_split = split;
			m_ptr   = p;
			m_perf  = 0.f;
		}
		void one_split_evaluator::operator()(){
			m_ptr->reset_params();
			m_ptr->switch_dataset(m_split,CM_TRAIN);
			m_ptr->fit();
			m_ptr->switch_dataset(m_split,CM_VALID);
			m_perf += m_ptr->predict();
		}

		all_splits_evaluator::all_splits_evaluator(boost::shared_ptr<crossvalidatable> p){
			m_ptr        = p;
			m_test_perf  = 0.f;
			m_test_perf0 = 0.f;
            m_perf       = 0.f;
		}
		float all_splits_evaluator::operator()(){
			for (unsigned int s = 0; s < m_ptr->n_splits(); ++s)
			{
                std::cout << "Processing split "<<s<<"/"<<m_ptr->n_splits()<<std::endl;
				m_ptr->switch_dataset(s,CM_TRAIN);
				m_ptr->fit();
				m_ptr->switch_dataset(s,CM_VALID);
				m_perf += m_ptr->predict();
                std::cout << "X-val error:" << m_perf/(s+1)  << std::endl;
                if(s < m_ptr->n_splits()-1)
                    m_ptr->reset_params(); // otherwise taken care of below
			}
			m_perf /= m_ptr->n_splits();

            if(m_ptr->refit_for_test()){
                if(m_perf < m_ptr->refit_thresh()){ // save time!
                    // retrain on TRAINALL (incl. VAL) and test on TEST
					std::cout << "Training on TRAINVAL..." << std::endl;
                    m_ptr->reset_params();
                    m_ptr->switch_dataset(0,CM_TRAINALL);
                    m_ptr->fit();
                }else{
					std::cout << "Skipping training on TRAINVAL: not good enough." << std::endl;
				}
                m_ptr->switch_dataset(0,CM_TEST);
                m_test_perf = m_ptr->predict();
                std::cout << "Test error:" << m_test_perf << std::endl;
            }else{
                // test last model on TEST w/o retraining
                m_ptr->switch_dataset(0,CM_TEST);
                m_test_perf0 = m_ptr->predict();
                std::cout << "Test0 error:" << m_test_perf0 << std::endl;
            }
            return m_perf; /* return the X-val error for optimization! */
		}

		void crossvalidation_queue::dispatch(boost::shared_ptr<crossvalidatable> p, const mongo::BSONObj& desc){
			namespace bar = boost::archive;
			std::ostringstream ss;
			{
				bar::binary_oarchive oa(ss);
				oa << p;
			}
			std::string s = ss.str();
			m_hub.insert_job(mongo::BSONObjBuilder()
					.append("task","cv_allsplits")
					.append("conf",desc)
					.appendBinData("cvable",s.size(),mongo::BinDataGeneral,s.c_str())
					.obj(), 3600 * 24); // 24 hours(??)
		}

		crossvalidation_worker::crossvalidation_worker(const std::string& url, const std::string& prefix)
			:mdbq::Client(url,prefix){
				g_worker = this;
			}

		void crossvalidation_worker::handle_task(const mongo::BSONObj& task){
			namespace bar = boost::archive;
			std::cout << "Handling task..."<<std::endl;
			if(task["task"].String() == "cv_allsplits"){
				boost::shared_ptr<crossvalidatable> p;
				int len;
				const char* cvable_ = task["cvable"].binData(len);
				std::string cvable(cvable_, len);
				std::istringstream ss(cvable);
				{
					bar::binary_iarchive ia(ss);
					ia >> p;
				}

				p->constructFromBSON(task["conf"].Obj());
				all_splits_evaluator ase(p);
                try{
                    ase();
                    this->finish(BSON("perf"<<ase.perf()<<"test_perf"<<ase.test_perf()<<"test_perf0"<<ase.test_perf0()));
                }catch(mdbq::timeout_exception& e){
                    // MDBQ already marked this as finished and failed!
                    throw;
                }catch(std::runtime_error& e){
                    this->finish(BSON("runtime_error"<<e.what()), 0);
                    throw;
                }
                throw std::runtime_error("finished successfully, but crashing to ensure memory is freed *cough*");
			}
		}
	}

}
