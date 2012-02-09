#ifndef __CUVNET_LEARNER_HPP__
#     define __CUVNET_LEARNER_HPP__

#include <datasets/dataset.hpp>
#include <datasets/splitter.hpp>

namespace mongo
{
    class BSONObj;
}
namespace cuvnet
{
    /**
     * helper to learn on simple datasets that fit in RAM (MNIST, CIFAR, etc)
     *
     * derived classes still need to imlpement:
     * - constructFromBSON (learner specific parts)
     * - before_batch (load batch data, but can use get_data_batch/get_label_batch from here)
     * - after_batch (accumulate performance data)
     * - before_epoch (reset performance data)
     * - after_epoch (log performance data)
     * - reset_params
     * - fit
     * - perf (get performance data)
     */
    template<class StorageSpace>
    class SimpleDatasetLearner
    {
        protected:
            /// Batch size
            unsigned int m_bs;
            /// the dataset with associated splits
            splitter     m_splits;
            /// true if we are should work on the validation set (e.g. for early stopping)
            bool         m_in_validation_mode;

            cuv::tensor<float,StorageSpace> m_current_data, m_current_vdata;
            cuv::tensor<float,StorageSpace> m_current_labels, m_current_vlabels;

        public:
            /**
             * create this object from a BSONObj
             */
            void constructFromBSON(const mongo::BSONObj& o);

            /**
             * switch to a split and a cross-validation mode
             */
            void switch_dataset(unsigned int split, cv_mode mode);

            void before_validation_epoch(); /// sets m_in_validation_mode to true
            void after_validation_epoch();  /// sets m_in_validation_mode to false
            
            /// @return a view to the requested data batch 
            cuv::tensor<float, StorageSpace> get_data_batch(unsigned int batch);

            /// @return a view to the requested label batch
            cuv::tensor<float, StorageSpace> get_label_batch(unsigned int batch);

            /// @return batch size
            inline unsigned int batchsize()const{return m_bs;}
            
            /// @return data dimension
            inline unsigned int datadim()const{ return m_splits.get_ds().train_data.shape(1); }

            /// @return the contained dataset (for meta-infos such as n-channels)
            inline const dataset& get_ds()const{ return m_splits.get_ds(); }

            /// @return the number of batches for current split+cv-mode
            unsigned int n_batches()const;
            
            /// @return the number of splits
            unsigned int n_splits()const{ return m_splits.n_splits(); }

            /// @returns whether early stopping can be used
            bool can_earlystop()const{ return m_current_vdata.ptr()!=NULL; }
    };

}

#endif /* __CUVNET_LEARNER_HPP__ */
