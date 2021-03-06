#include <mongo/bson/bson.h>
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
#include <cuvnet/op.hpp>
#include "learner.hpp"
namespace cuvnet
{

    /**
     * split training set items, but assume that each one is stored in a /bag/,
     * which is completely either in a split or not.
     */
    cuv::tensor<int,cuv::host_memory_space> 
        split_bags(unsigned int n_splits, const cuv::tensor<int,cuv::host_memory_space>& bag_index, float frac_val = 0.2f){
            cuv::tensor<int,cuv::host_memory_space> split_index(bag_index.shape());

            unsigned int n_bags = cuv::maximum(bag_index) + 1; // +1: bag number 0

            // create a mapping from a bag id to a random new bag id
            std::vector<int> rnd_bags (n_bags);
            for (unsigned int i = 0; i < n_bags; ++i)
                rnd_bags[i] = i;
            std::random_shuffle(rnd_bags.begin(),rnd_bags.end());

            // now each bag is in fold `i' when its random new bag id modulo n_splits is `i'.
            if(n_splits > 1){
                for(unsigned int i=0;i<bag_index.size();i++)
                    split_index[i] = rnd_bags[bag_index[i]] % n_splits;
            }else{
                for(unsigned int i=0;i<bag_index.size();i++)
                    split_index[i] = rnd_bags[bag_index[i]] / (float) n_bags > frac_val; // larger fraction goes in split `1'
            }
            return split_index;
    }

    template<class StorageSpace>
    std::string SimpleDatasetLearner<StorageSpace>::describe_current_mode_split(bool verbose){
        std::string desc;
        if(verbose){
            desc = " patterns: "+boost::lexical_cast<std::string>(n_batches()*m_bs);
        }
        if(m_current_mode == CM_TRAINALL)
            return "TRAINALL"+desc;
        if(m_current_mode == CM_TRAIN)
            return "TRAIN_SPLIT" + boost::lexical_cast<std::string>(m_current_split)+desc;
        if(m_current_mode == CM_VALID)
            return "VALID_SPLIT" + boost::lexical_cast<std::string>(m_current_split)+desc;
        if(m_current_mode == CM_TEST)
            return "TEST"+desc;
        throw std::runtime_error("SimpleDatasetLearner: Unknown mode!");
    }

    template<class StorageSpace>
    void SimpleDatasetLearner<StorageSpace>::switch_dataset(unsigned int split, cv_mode mode){
        cuv::tensor<float,cuv::host_memory_space> data, vdata;
        cuv::tensor<float,  cuv::host_memory_space> labels, vlabels;

        m_current_mode  = mode;
        m_current_split = split;

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
    void SimpleDatasetLearner<StorageSpace>::init(int bs, std::string ds, unsigned int nsplits, float es_frac){
        m_ds_name = ds;
        m_bs = bs;
        m_in_early_stopping = false;
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
            randomizer().transform(dsall.train_data, dsall.train_labels);
            if(from < to)
                dsall = select_training_range(dsall, from, to);
            // no transformation except randomization needed
            m_splits.init(dsall, nsplits);
        }else if(ds == "msrc_descr"){
            msrc_desc_dataset dsall("/home/local/datasets/msrc_superpixel");
            randomizer().transform(dsall.train_data, dsall.train_labels, &dsall.imagen);


            //zero_mean_unit_variance<> zmuv;
            //zmuv.fit_transform(dsall.train_data);
            //zmuv.transform(dsall.test_data);

            // split images, not descriptors, since descriptors contain global
            // (image) descriptor and are therefore not really i.i.d!
            cuv::tensor<int,cuv::host_memory_space> splits = split_bags(nsplits, dsall.imagen);
            m_splits.init(dsall, splits);
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
    template<class StorageSpace>
    cuv::tensor<float, StorageSpace> 
    SimpleDatasetLearner<StorageSpace>::get_data_batch(unsigned int batch){
        cuv::tensor<float,StorageSpace>& data = m_in_early_stopping ? m_current_vdata : m_current_data;
        return data[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
    }
    template<class StorageSpace>
    cuv::tensor<float, StorageSpace> 
    SimpleDatasetLearner<StorageSpace>::get_label_batch(unsigned int batch){
        cuv::tensor<float,StorageSpace>& labl = m_in_early_stopping ? m_current_vlabels : m_current_labels;
        return labl[cuv::indices[cuv::index_range(batch*m_bs,(batch+1)*m_bs)][cuv::index_range()]];
    }

    template<class StorageSpace>
    void
    SimpleDatasetLearner<StorageSpace>::before_early_stopping_epoch(){ 
        m_in_early_stopping = true; 
    }

    template<class StorageSpace>
    void
    SimpleDatasetLearner<StorageSpace>::after_early_stopping_epoch(){ 
        m_in_early_stopping = false; 
    }

    template<class StorageSpace>
    unsigned int
    SimpleDatasetLearner<StorageSpace>::n_batches()const{ 
        if(m_in_early_stopping){
            return m_current_vdata.shape(0)/m_bs;
        }else{
            return m_current_data.shape(0)/m_bs;
        }
    }

    template<class StorageSpace>
    bool 
    SimpleDatasetLearner<StorageSpace>::can_earlystop()const{ 
        return m_current_vdata.ptr();
    }

    template class SimpleDatasetLearner<cuv::dev_memory_space>;
    template class SimpleDatasetLearner<cuv::host_memory_space>;
}
