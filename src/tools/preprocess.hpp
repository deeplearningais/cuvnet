#ifndef __CUVNET_PREPROCESS_HPP__
#     define __CUVNET_PREPROCESS_HPP__

#include<cuv.hpp>

namespace cuvnet
{
    class preprocessor{
        public:
            virtual void process_filename(const std::string& fn)=0;
            virtual void process_filestring(const std::string& fn)=0;
    };

    class patch_extractor
    : public preprocessor
    {
        private:
        public:
            void process_filename(const std::string& fn);
            void process_filestring(const std::string& fn);
    };
}
#endif /* __CUVNET_PREPROCESS_HPP__ */
