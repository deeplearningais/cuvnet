#ifndef __NPY_DATASETS_HPP__
#     define __NPY_DATASETS_HPP__

#include "dataset.hpp"

namespace cuvnet
{
    /**
     * Provides access to datasets stored as numpy matrices (stored with numpy.save()).
     *
     * @ingroup datasets
     */
    struct npy_dataset
        : public dataset
    {
        public:
            npy_dataset(const std::string& path);
    };

}
#endif /* __NPY_DATASETS_HPP__ */
