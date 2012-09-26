#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/functional/hash.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cuv/basics/io.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>

#include "read_amat.hpp"
#include "amat_datasets.hpp"

namespace cuvnet
{
    bool amat_dataset::read_cached(const std::string& zipfile, const std::string& train, const std::string& test){
        boost::hash<std::string> hasher;
        std::string fn = "/tmp/" + boost::lexical_cast<std::string>(hasher(zipfile + train + test)) + ".cache";
        std::ifstream is(fn.c_str());
        if(!is.good())
            return false;
        boost::archive::binary_iarchive ia(is);
        ia >> train_data >> train_labels >> val_data >> val_labels >> test_data >> test_labels;
        return true;
    }
    void amat_dataset::store_cache(const std::string& zipfile, const std::string& train, const std::string& test){
        boost::hash<std::string> hasher;
        std::string fn = "/tmp/" + boost::lexical_cast<std::string>(hasher(zipfile + train + test)) + ".cache";
        std::cout << "Caching " << zipfile<< " in " << fn << std::endl;

        std::ofstream os(fn.c_str());
        boost::archive::binary_oarchive oa(os);

        oa << train_data << train_labels << val_data << val_labels << test_data << test_labels;
    }
    amat_dataset::amat_dataset(const std::string& zipfile, const std::string& train, const std::string& test){
        unsigned int n_classes = 10;
        if(zipfile.find("convex") != std::string::npos)
            n_classes = 2;
        channels = 1;
        binary   = true;
        image_size = 28;

        bool cache_success = read_cached(zipfile,train,test);
        if(cache_success)
            return;

        cuv::tensor<int, cuv::host_memory_space> trainl;
        cuv::tensor<int, cuv::host_memory_space> testl;

        read_amat_with_label(train_data,trainl,zipfile,train);
        read_amat_with_label(test_data,testl,zipfile,test);

        train_labels.resize(cuv::extents[train_data.shape(0)][n_classes]);
        test_labels.resize(cuv::extents[test_data.shape(0)][n_classes]);
        train_labels = 0.f;
        test_labels = 0.f;
        for (unsigned int i = 0; i < trainl.size(); ++i){
            train_labels(i, trainl[i]) = 1;
        }
        for (unsigned int i = 0; i < testl.size(); ++i){
            test_labels(i, testl[i]) = 1;
        }

        std::cout << "read amat with train_data shape "<<train_data.info().host_shape<<std::endl;
        std::cout << "                test_data shape "<<test_data.info().host_shape<<std::endl;
        std::cout << "                train_lbl shape "<<train_labels.info().host_shape<<std::endl;
        std::cout << "                 test_lbl shape "<<test_labels.info().host_shape<<std::endl;
        store_cache(zipfile, train, test);
    }
    
}
