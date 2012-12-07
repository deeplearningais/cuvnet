#ifndef __CUVNET_LEARNER_HPP__
#     define __CUVNET_LEARNER_HPP__

#include <datasets/dataset.hpp>
#include <datasets/splitter.hpp>
#include <boost/serialization/split_member.hpp>

namespace cuvnet
{
    /**
     * helper to learn on simple datasets that fit in RAM (MNIST, CIFAR, etc).
     *
     * derived classes still need to imlpement:
     * - constructFromBSON (learner specific parts)
     * - before_batch (load batch data, but can use get_data_batch/get_label_batch from here)
     * - after_batch (accumulate performance data)
     * - before_epoch (reset performance data)
     * - after_epoch (log performance data)
     * - reset_params
     * - fit (train model)
     * - perf (get performance data)
     *
     *   @ingroup tools
     */
    template<class StorageSpace>
    class SimpleDatasetLearner
    {
    private:
        friend class boost::serialization::access;

        template<class Archive>
            void save(Archive & ar, const unsigned int version) const
            {
                unsigned int n_splits = m_splits.size();
                ar << m_bs << m_ds_name << n_splits;
            }

        template<class Archive>
            void load(Archive & ar, const unsigned int version)
            {
                unsigned int n_splits;
                ar >> m_bs >> m_ds_name >> n_splits;
                init(m_bs, m_ds_name, n_splits);
            }

        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                boost::serialization::split_member(ar, *this, version);
            }
    public:
        protected:
            /// Batch size
            unsigned int m_bs;
            /// the dataset with associated splits
            splitter     m_splits;
            /// true if we are evaluating for early stopping
            bool         m_in_early_stopping;

            /// dataset name for de-serialization
            std::string m_ds_name;

            /// the current cv_mode
            cv_mode      m_current_mode;
            /// the current split
            unsigned int m_current_split;

            cuv::tensor<float,StorageSpace> m_current_data, m_current_vdata;
            cuv::tensor<float,StorageSpace> m_current_labels, m_current_vlabels;

        public:

            /**
             * initialize 
             */
            void init(int bs, std::string ds, unsigned int nsplits, float es_frac=0.1f);
            
            /**
             * switch to a split and a cross-validation mode
             */
            void switch_dataset(unsigned int split, cv_mode mode);

            /// @return the current cv-mode
            inline cv_mode get_current_cv_mode(){ return m_current_mode; }

            /// @return the current cv-mode
            inline unsigned int get_current_split(){ return m_current_split; }

            /// @return a string describing the current mode/split
            std::string describe_current_mode_split(bool verbose);

            void before_early_stopping_epoch(); ///< sets m_in_validation_mode to true
            void after_early_stopping_epoch();  ///< sets m_in_validation_mode to false
            
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
            bool can_earlystop()const;

			/// set the batchsize
			inline void set_batchsize(unsigned int bs){m_bs=bs;}
    };

}

#endif /* __CUVNET_LEARNER_HPP__ */
