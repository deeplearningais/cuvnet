#include <cuv/convert/convert.hpp>
#include <datasets/mnist.hpp>
#include <datasets/cifar.hpp>
#include <datasets/ldpc.hpp>
#include <datasets/natural.hpp>
#include <datasets/tiny_mnist.hpp>
#include <datasets/randomizer.hpp>
#include <datasets/amat_datasets.hpp>
#include <datasets/npy_datasets.hpp>
#include <datasets/msrc_descriptors.hpp>
#include <datasets/splitter.hpp>
#include <cuvnet/tools/preprocess.hpp>
#include <cuvnet/tools/logging.hpp>
#include <cuvnet/op.hpp>

#include "simple_dataset_learner.hpp"

namespace 
{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("simple_ds_lrn"));
}

namespace cuvnet
{
    simple_dataset_learner::simple_dataset_learner()
        :   m_current_mode(CM_TRAIN)
        ,   m_current_split(0)
    {}
    void 
        simple_dataset_learner::init(std::string ds, unsigned int nsplits, float es_frac){
            size_t pos = ds.find('?');
            unsigned int from = INT_MAX, to = 0; // inverse range == not used
            if(pos != std::string::npos){
                std::string ds0 = ds.substr(0, pos);
                std::string subsetspec = ds.substr(pos+1);
                size_t colonpos = subsetspec.find(':');
                if(colonpos == std::string::npos){
                    throw std::runtime_error("Could not parse dataset string, should have format 'dataset?from:to'");
                }
                from = boost::lexical_cast<int>(subsetspec.substr(0,colonpos));
                to   = boost::lexical_cast<int>(subsetspec.substr(colonpos+1));
                ds = ds0;
            }
            if(0){
                ;
            }else if (ds == "tiny_mnist"){
                dataset dsall = tiny_mnist("/home/local/datasets/tapani");
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                // the dataset is already shuffled and normalized for zero-mean and unit variance.
                m_splits.init(dsall, nsplits, es_frac);
            }else if (ds == "mnist"){
                dataset dsall = mnist_dataset("/home/local/datasets/MNIST");
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                global_min_max_normalize<> normalizer(0,1);
                normalizer.fit_transform(dsall.train_data);
                normalizer.transform(dsall.test_data);
                m_splits.init(dsall, nsplits, es_frac);
            }else if (ds == "convex"){
                dataset dsall = amat_dataset("/home/local/datasets/bengio/convex.zip", "convex_train.amat","50k/convex_test.amat");
                dsall.binary = true; // Note: not all amat_datasets are binary!
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                //randomizer().transform(dsall.train_data, dsall.train_labels);
                global_min_max_normalize<> normalizer(0,1);
                normalizer.fit_transform(dsall.train_data);
                normalizer.transform(dsall.test_data);
                m_splits.init(dsall, nsplits, es_frac); // es_frac=0.25 --> trainval: 8000, train:6000, val: 2000
            }else if (ds == "mnist_rot"){
                dataset dsall = amat_dataset("/home/local/datasets/bengio/mnist_rotation_new.zip", "mnist_all_rotation_normalized_float_train_valid.amat","mnist_all_rotation_normalized_float_test.amat");
                dsall.binary = true; // Note: not all amat_datasets are binary!
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                //randomizer().transform(dsall.train_data, dsall.train_labels);
                global_min_max_normalize<> normalizer(0,1);
                normalizer.fit_transform(dsall.train_data);
                normalizer.transform(dsall.test_data);
                m_splits.init(dsall, nsplits, es_frac);
            }else if (ds == "ldpc"){
                dataset dsall = ldpc_dataset("/home/local/datasets/LDPC");
                //randomizer().transform(dsall.train_data, dsall.train_labels); // ds is already shuffled when generating!
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                // no transformation except randomization needed
                m_splits.init(dsall, nsplits);
            }else if (ds == "natural"){
                dataset dsall = natural_dataset("/home/local/datasets/natural_images");
                // TODO: this one has complicated pre-processing, needs normalizer
                // to be accessible for filter visualization
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                {
                    // after applying logarithm, data distribution looks roughly gaussian.
                    log_transformer<cuv::host_memory_space> lt;
                    lt.fit_transform(dsall.train_data);
                    lt.transform(dsall.test_data);
                }
                /*
                 *{
                 *    global_min_max_normalize<> normalizer(0,1);
                 *    normalizer.fit_transform(dsall.train_data);
                 *    normalizer.transform(dsall.test_data);
                 *}
                 */
                {
                    zero_sample_mean<> zsm;
                    zsm.fit_transform(dsall.train_data);
                    zsm.transform(dsall.test_data);
                }
                {
                    zero_mean_unit_variance<> zmuv;
                    zmuv.fit_transform(dsall.train_data);
                    zmuv.transform(dsall.test_data);
                }
                m_splits.init(dsall, nsplits, es_frac);
            }else if(ds == "cifar_zca"){
                dataset dsall = npy_dataset("/home/local/datasets/cifar10/pylearn2_gcn_whitened");
                dsall.channels = 3;
                dsall.image_size = 32;
                dsall.binary = false;
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                m_splits.init(dsall, nsplits, es_frac);
            }else if(ds == "cifar"){
                dataset dsall = cifar_dataset("/home/local/datasets/CIFAR10");
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                zero_mean_unit_variance<> normalizer;
                normalizer.fit_transform(dsall.train_data);
                normalizer.transform(dsall.test_data);
                m_splits.init(dsall, nsplits, es_frac);
            }else if(ds == "cifar_gray"){
                dataset dsall = cifar_dataset("/home/local/datasets/CIFAR10", true);
                randomizer().transform(dsall.train_data, dsall.train_labels);
                if(from < to)
                    dsall = select_training_range(dsall, from, to);
                zero_mean_unit_variance<> normalizer;
                normalizer.fit_transform(dsall.train_data);
                normalizer.transform(dsall.test_data);
                m_splits.init(dsall, nsplits, es_frac);
            }else {
                throw std::runtime_error("unknown dataset `"+ds+"'");
            }
        }

    void 
        simple_dataset_learner::switch_dataset(cv_mode mode, int split){
            cuv::tensor<float,cuv::host_memory_space> data, vdata;
            cuv::tensor<float,  cuv::host_memory_space> labels, vlabels;

            m_current_mode  = mode;
            if(split >= 0)
                m_current_split = split;

            dataset ds = m_splits[m_current_split];
            switch(mode) {
                case CM_TRAINALL:
                    data   = m_splits.get_ds().train_data;
                    labels = m_splits.get_ds().train_labels;
                    break;
                case CM_TRAIN:
                    data    = ds.train_data;
                    labels  = ds.train_labels;
                    vdata   = ds.val_data;      // for early stopping!
                    vlabels = ds.val_labels;    // for early stopping!
                    break;
                case CM_VALID:
                    vdata   = ds.val_data;
                    vlabels = ds.val_labels;
                    break;
                case CM_TEST:
                    data   = ds.test_data;
                    labels = ds.test_labels;
                    break;
            };
            // convert labels to float
            if(labels.ndim()){
                host_matrix flabels(labels.shape());
                cuv::convert(flabels,  labels);
                m_current_data    = data;
                m_current_labels  = flabels;
            }
            if(vlabels.ndim()) {
                host_matrix fvlabels(vlabels.shape());
                cuv::convert(fvlabels, vlabels);
                m_current_vdata   = vdata;
                m_current_vlabels = fvlabels;
            } else {
                m_current_vdata.dealloc();
                m_current_vlabels.dealloc();
            }
        }

    unsigned int 
        simple_dataset_learner::n_batches(unsigned int batchsize){ 
            if(m_current_mode == CM_VALID)
                return m_current_vdata.shape(0)/batchsize;
            return m_current_data.shape(0)/batchsize;
        }

    void 
        simple_dataset_learner::load_batch(model* m, unsigned int epoch, unsigned int bid){
            std::vector<Op*> inputs = m->get_inputs();
            assert(inputs.size() == 2 || inputs.size() == 1);

            ParameterInput *X = NULL, *Y = NULL;

            X = dynamic_cast<ParameterInput*>(inputs[0]);
            assert(X);
            if(inputs.size() == 2){
                Y = dynamic_cast<ParameterInput*>(inputs[1]);
                assert(Y);
            }

            unsigned int bs = X->data().shape(0);
#ifndef NDEBUG
            if(Y){
                unsigned int bs2 = Y->data().shape(0);
                assert(bs == bs2);
            }
#endif
            bool in_early_stopping = m_current_mode == CM_VALID;
            {
                host_matrix& data = in_early_stopping ? m_current_vdata : m_current_data;
                X->data() = data[cuv::indices[cuv::index_range(bid*bs,(bid+1)*bs)]];
            }
            if(Y){
                host_matrix& labl = in_early_stopping ? m_current_vlabels : m_current_labels;
                Y->data() = labl[cuv::indices[cuv::index_range(bid*bs,(bid+1)*bs)]];
            }
        }
    
    host_matrix& simple_dataset_learner::get_current_data()
    {
        bool in_early_stopping = m_current_mode == CM_VALID;
        host_matrix& data = in_early_stopping ? m_current_vdata : m_current_data;
        return data;
    }

    host_matrix& simple_dataset_learner::get_current_labels()
    {
        bool in_early_stopping = m_current_mode == CM_VALID;
        host_matrix& labels = in_early_stopping ? m_current_vlabels : m_current_labels;
        return labels;
    }

    unsigned int simple_dataset_learner::size()
    {
        return get_current_labels().shape(0);
    }
}
