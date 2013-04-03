#ifndef __CROSSVALID_HPP__
#     define __CROSSVALID_HPP__
#include <boost/asio.hpp>
//#include <boost/asio/signal_set.hpp>
//#include <boost/limits.hpp>
//#include <boost/bind.hpp>


#include <datasets/dataset.hpp>
#include <mdbq/hub.hpp>
#include <mdbq/client.hpp>

namespace cuvnet
{
    struct cuda_thread
    {
        boost::asio::io_service& m_io_service;
        int m_dev;
        cuda_thread(boost::asio::io_service& s, int dev):m_io_service(s),m_dev(dev){ }
        void operator()();
    };

	/** 
     * to use crossvalidation, you need to derive from this
     * @ingroup learning
     */
	class crossvalidatable{
		private:
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { }
		public:
			/// switch  to a different split/crossvalidation mode
			virtual void switch_dataset(unsigned int split, cv_mode mode)=0;

			/// learn on current split
			virtual void fit()=0;

			/// predict on current test set
			virtual float predict()=0;

			/// reset parameters
			virtual void reset_params()=0;

			/// return the number of requested splits
			virtual unsigned int n_splits()=0;

			/// return minimum on VAL before retraining for TEST is performed
            /// @return INT_MAX
			virtual float refit_thresh()const;

            /// should return true if retrain for fit is required
            /// @return true
            virtual bool refit_for_test()const;
	};

    /**
     * a simpler version of crossvalidatable, when only a validation set is
     * used (no real cross-validation).
     * @ingroup learning
     */
    class validatable{
		private:
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) { }
		public:
			/** learn on current split.
             * This method should set the performance member variables.
             */
			virtual void fit()=0;

			/// predict on current test set
			virtual float predict()=0;

			/// reset parameters
			virtual void reset_params()=0;

            /// access performance on validation set after training on TRAIN
            virtual float val_perf()const{ return INT_MAX; }

            /// access performance on TEST after training on TRAIN
            virtual float test_perf0()const{ return INT_MAX; }

            /// access performance on TEST after training on TRAINVAL
            virtual float test_perf()const{ return INT_MAX; }
    };

    /**
     * crossvalidation namespace
     * @ingroup learning
     */
	namespace cv{

		/**
		 * evaluates exactly one split
         * @ingroup learning
		 */
		struct one_split_evaluator{
			public:
				one_split_evaluator(unsigned int split, boost::shared_ptr<crossvalidatable> p);
				void operator()();

				inline float perf()const{return m_perf;}
			private:
				unsigned int m_split;
				float m_perf;
				boost::shared_ptr<crossvalidatable> m_ptr;
		};

		/**
		 * evaluates on all splits
         * @ingroup learning
		 */
		struct all_splits_evaluator{
			public:
				all_splits_evaluator(boost::shared_ptr<crossvalidatable> p);
				float operator()();

                /// @return the average CV performance
				inline float perf()const{return m_perf;}

                /// @return the test performance after training on TRAINVAL
				inline float test_perf()const{return m_test_perf;}

                /// @return the test performance of last CV model (only really
                ///useful if only one TRAIN/VAL split is used)
				inline float test_perf0()const{return m_test_perf0;}
			private:
				float m_perf;       ///< VAL performance after training on TRAIN
                float m_test_perf0; ///< test performance of model trained only on TRAIN
				float m_test_perf;  ///< test performance after training on TRAINVAL
				boost::shared_ptr<crossvalidatable> m_ptr;
		};

		/**
		 * Crossvalidation Queue
         * @ingroup learning
		 */
		struct crossvalidation_queue{
			mdbq::Hub m_hub;
			crossvalidation_queue(const std::string& url, const std::string& prefix):m_hub(url,prefix){}
			void dispatch(boost::shared_ptr<crossvalidatable> p, const mongo::BSONObj& description);
		};

		/**
		 * Crossvalidation Worker
         * @ingroup learning
		 */
		struct crossvalidation_worker
		: mdbq::Client{
			crossvalidation_worker(const std::string& url, const std::string& prefix);
			void handle_task(const mongo::BSONObj& task);
		};
	}

}
#endif /* __CROSSVALID_HPP__ */
