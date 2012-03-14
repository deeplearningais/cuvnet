#include <mongo/bson/bson.h>
#include <cuv/convert/convert.hpp>
#include <datasets/mnist.hpp>
#include <datasets/cifar.hpp>
#include <datasets/ldpc.hpp>
#include <datasets/natural.hpp>
#include <datasets/randomizer.hpp>
#include <datasets/amat_datasets.hpp>
#include <tools/preprocess.hpp>
#include <cuvnet/op.hpp>
#include "learner.hpp"
namespace cuvnet
{

    template<class StorageSpace>
    void SimpleDatasetLearner<StorageSpace>::switch_dataset(unsigned int split, cv_mode mode){
        cuv::tensor<float,cuv::host_memory_space> data, vdata;
        cuv::tensor<int,  cuv::host_memory_space> labels, vlabels;
        dataset ds = m_splits[split];
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
                data   = ds.val_data;
                labels = ds.val_labels;
                break;
            case CM_TEST:
                data   = ds.test_data;
                labels = ds.test_labels;
                break;
        };

        // convert labels to float
        cuv::tensor<float,cuv::host_memory_space> flabels(labels.shape());
        cuv::convert(flabels,  labels);
        m_current_data    = data;
        m_current_labels  = flabels;
        if(vlabels.ndim()) {
            cuv::tensor<float,cuv::host_memory_space> fvlabels(vlabels.shape());
            cuv::convert(fvlabels, vlabels);
            m_current_vdata   = vdata;
            m_current_vlabels = fvlabels;
        } else {
            m_current_vdata.dealloc();
            m_current_vlabels.dealloc();
        }
    }

    template<class StorageSpace>
    void SimpleDatasetLearner<StorageSpace>::constructFromBSON(const mongo::BSONObj& o){
        m_bs                 = o["bs"].Int();
        std::string ds       = o["dataset"].String();
        unsigned int nsplits = o["nsplits"].Int();
        if(0);
        else if (ds == "mnist"){
            dataset dsall = mnist_dataset("/home/local/datasets/MNIST");
            dsall         = randomizer().transform(dsall);
            global_min_max_normalize<> normalizer(0,1);
            normalizer.fit_transform(dsall.train_data);
            normalizer.transform(dsall.test_data);
            m_splits.init(dsall, nsplits);
        }else if (ds == "mnist_rot"){
            dataset dsall = amat_dataset("/home/local/datasets/bengio/mnist_rotation_new.zip", "mnist_all_rotation_normalized_float_train_valid.amat","mnist_all_rotation_normalized_float_test.amat");
            dsall.binary = true; // Note: not all amat_datasets are binary!
            dsall         = randomizer().transform(dsall);
            global_min_max_normalize<> normalizer(0,1);
            normalizer.fit_transform(dsall.train_data);
            normalizer.transform(dsall.test_data);
            m_splits.init(dsall, nsplits);
        }else if (ds == "ldpc"){
            dataset dsall = ldpc_dataset("/home/local/datasets/LDPC");
            dsall         = randomizer().transform(dsall);
            // no transformation except randomization needed
            m_splits.init(dsall, nsplits);
        }else if (ds == "natural"){
            dataset dsall = natural_dataset("/home/local/datasets/natural_images");
            // TODO: this one has complicated pre-processing, needs normalizer
            // to be accessible for filter visualization
            cuvAssert(false);
            dsall         = randomizer().transform(dsall);
            global_min_max_normalize<> normalizer(0,1);
            normalizer.fit_transform(dsall.train_data);
            normalizer.transform(dsall.test_data);
            m_splits.init(dsall, nsplits);
        }else if(ds == "cifar"){
            dataset dsall = cifar_dataset();
            dsall = randomizer().transform(dsall);
            zero_mean_unit_variance<> normalizer;
            normalizer.fit_transform(dsall.train_data);
            normalizer.transform(dsall.test_data);
            m_splits.init(dsall, nsplits);
        }else {
            throw std::runtime_error("unknown dataset `"+ds+"'");
        }
    }
    template<class StorageSpace>
    cuv::tensor<float, StorageSpace> 
    SimpleDatasetLearner<StorageSpace>::get_data_batch(unsigned int batch){
        cuv::tensor<float,StorageSpace>& data = m_in_validation_mode ? m_current_vdata : m_current_data;
        return data[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
    }
    template<class StorageSpace>
    cuv::tensor<float, StorageSpace> 
    SimpleDatasetLearner<StorageSpace>::get_label_batch(unsigned int batch){
        cuv::tensor<float,StorageSpace>& labl = m_in_validation_mode ? m_current_vlabels : m_current_labels;
        return labl[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
    }

    template<class StorageSpace>
    void
    SimpleDatasetLearner<StorageSpace>::before_validation_epoch(){ m_in_validation_mode = true; }

    template<class StorageSpace>
    void
    SimpleDatasetLearner<StorageSpace>::after_validation_epoch(){ m_in_validation_mode = false; }

    template<class StorageSpace>
    unsigned int
    SimpleDatasetLearner<StorageSpace>::n_batches()const{ 
        return  m_in_validation_mode ?
            m_current_vdata.shape(0)/m_bs :
            m_current_data.shape(0)/m_bs;
    }

    template class SimpleDatasetLearner<cuv::dev_memory_space>;
    template class SimpleDatasetLearner<cuv::host_memory_space>;
}
