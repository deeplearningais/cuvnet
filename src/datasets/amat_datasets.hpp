#ifndef __AMAT_DATASETS_HPP__
#     define __AMAT_DATASETS_HPP__

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
        private:
            bool read_cached(const std::string& zipfile, const std::string& train, const std::string& test);
            void store_cache(const std::string& zipfile, const std::string& train, const std::string& test);
        public:
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
        amat_dataset(const std::string& zipfile, const std::string& train, const std::string& test);
    };

}
#endif /* __AMAT_DATASETS_HPP__ */
