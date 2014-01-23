#include <cuv.hpp>
#include "smart_ptr.hpp"

namespace cuvnet
{
    boost::shared_ptr<cuv::allocator>
        get_global_allocator(){
            // NOTE we cannot use ordinary logging here, since this
            // is used to set up a static variable which is initialized before
            // logging is set up properly.
            //std::cout << "Creating pooled_cuda_allocator!"<< std::endl;
            return boost::make_shared<cuv::pooled_cuda_allocator>("cow_ptr");
            //return boost::make_shared<cuv::default_allocator>();
        }
}
