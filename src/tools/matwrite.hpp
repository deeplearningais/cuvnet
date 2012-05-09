#include <cuv/basics/tensor.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <third_party/libnpy-0.5/include/npy.h>

namespace cuvnet
{
    template<class T>
    void tofile(const std::string& fn, const cuv::tensor<T,cuv::host_memory_space>& mat){
        std::vector<int> shape(mat.ndim());

        std::copy(
                &mat.info().host_shape[0],
                &mat.info().host_shape[0]+mat.ndim(),
                shape.begin());
        if(cuv::IsSame<T,float>::Result::value){
            npy_save(const_cast<char*>(fn.c_str()), (char*)"float32", false, shape.size(), &shape[0], sizeof(float), (void*)mat.ptr());
        }else if(cuv::IsSame<T,int>::Result::value){
            npy_save(const_cast<char*>(fn.c_str()), (char*)"int32", false, shape.size(), &shape[0], sizeof(int), (void*)mat.ptr());
        }else{
            throw std::runtime_error("unknown tensor type");
        }
    }
    template<class T>
    void tofile(const std::string& fn, const cuv::tensor<T,cuv::dev_memory_space>& mat){
        cuv::tensor<T,cuv::host_memory_space> m = mat;
        tofile(fn,m);
    }
}
