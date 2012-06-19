#ifndef __AMAT_DATASETS_HPP__
#     define __AMAT_DATASETS_HPP__

#include <tools/read_amat.hpp>
#include <cuv/basics/io.hpp>
#include "dataset.hpp"

namespace cuvnet
{
    /**
     * Provides access to datasets stored in the zipped amat-Format as used in the
     * ICML 2007 paper by Larochelle et al.
     *
     * Title: An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation.
     * @ingroup datasets
     */
    struct amat_dataset
        : public dataset
    {
        /**
         * ctor.
         *
         * @note this ctor makes an assumption about the number of classes. It
         *       is 10, except if the name contains "convex", then it is 2.
         *
         * @param zipfile the name of the zipfile containing the data files
         * @param train name of the file in the zipfile containing train data
         * @param train name of the file in the zipfile containing test data
         */
        amat_dataset(const std::string& zipfile, const std::string& train, const std::string& test){
            cuv::tensor<int, cuv::host_memory_space> trainl;
            cuv::tensor<int, cuv::host_memory_space> testl;

            read_amat_with_label(train_data,trainl,zipfile,train);
            read_amat_with_label(test_data,testl,zipfile,test);



            unsigned int n_classes = 10;
            if(zipfile.find("convex") != std::string::npos)
                n_classes = 2;
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
            channels = 1;
            binary   = true;
            image_size = 28;
        }
    };

}
#endif /* __AMAT_DATASETS_HPP__ */
