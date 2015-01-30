#ifndef __CV_DATASETS_HPP__
#     define __CV_DATASETS_HPP__
#include "pattern_set.hpp"

namespace datasets{
    struct dataset{
        void load_batch(model*)=0;
    };
}

#endif /* __CV_DATASETS_HPP__ */
