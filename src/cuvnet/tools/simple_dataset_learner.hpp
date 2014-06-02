#ifndef __SIMPLE_DATASET_LEARNER_HPP__
#     define __SIMPLE_DATASET_LEARNER_HPP__
#include <datasets/dataset.hpp>
#include <datasets/splitter.hpp>
#include <cuvnet/tools/learner2.hpp>

namespace cuvnet
{
    class simple_dataset_learner
    : public learner2
    {
        private:
            splitter m_splits;
            cv_mode m_current_mode;
            unsigned int m_current_split;

            host_matrix m_current_data, m_current_vdata;
            host_matrix m_current_labels, m_current_vlabels;

        public:

            /**
             * ctor.
             */
            simple_dataset_learner();

            /**
             * load and split the dataset.
             * @param ds the name of the dataset
             * @param nsplits the number of splits for crossvalidation
             * @param es_frac the fraction of the dataset used for early-stopping if nsplits=1
             */
            void init(std::string ds, unsigned int nsplits, float es_frac=.1f);

            /**
             * switch to a split and cross-validate mode.
             * @param split the number of the split, -1 if unchanged
             * @param cv_mode the current mode (train, validation, test, trainall)
             */
            void switch_dataset(cv_mode mode, int split=-1);

            /**
             * determine the number of batched in the current split/mode.
             * @param batchsize the number of patterns loaded at once
             * @return the number of batches in the dataset
             */
            unsigned int n_batches(unsigned int batchsize);

            /**
             * @return the number of splits (as supplied in the init() function)
             */
            inline unsigned int n_splits()const{ return m_splits.n_splits(); }

            /**
             * load a batch from the current split/mode into a model.
             */
            void load_batch(model* m, unsigned int epoch, unsigned int bid);

            /**
             * @return the splitter object.
             */
            inline const splitter& get_splitter()const{ return m_splits; }
            
            /**
             * @return the current cv_mode.
             */
            inline const cv_mode& get_current_mode()const{return m_current_mode; }
            
            /**
             * @return the current data host_matrix.
             */
            host_matrix& get_current_data();

            /**
             * @return the current labels host_matrix.
             */
            host_matrix& get_current_labels();
    };
}
#endif /* __SIMPLE_DATASET_LEARNER_HPP__ */
