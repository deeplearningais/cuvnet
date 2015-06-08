#ifndef __MATWRITE_HPP__
#     define __MATWRITE_HPP__

#include <cuv/basics/tensor.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <third_party/libnpy-0.5/include/npy.h>
#include <third_party/cnpy/cnpy.h>

namespace cuvnet
{
    /**
     * dump a tensor to a numpy array on disk for analysis in python.
     * @ingroup tools
     */
    template<class T>
    void tofile(const std::string& fn, const cuv::tensor<T,cuv::host_memory_space>& mat){
        std::vector<int> shape(mat.ndim());

        std::copy(
                &mat.info().host_shape[0],
                &mat.info().host_shape[0]+mat.ndim(),
                shape.begin());
        if(cuv::IsSame<T,float>::Result::value){
            npy_save(const_cast<char*>(fn.c_str()), (char*)"<f4", false, shape.size(), &shape[0], sizeof(float), (void*)mat.ptr());
        }else if(cuv::IsSame<T,int>::Result::value){
            npy_save(const_cast<char*>(fn.c_str()), (char*)"<i4", false, shape.size(), &shape[0], sizeof(int), (void*)mat.ptr());
        }else{
            throw std::runtime_error("unknown tensor type");
        }
    }
    /**
     * dump a tensor to a numpy array on disk for analysis in python.
     * @overload
     * @ingroup tools
     */
    template<class T>
    void tofile(const std::string& fn, const cuv::tensor<T,cuv::dev_memory_space>& mat){
        cuv::tensor<T,cuv::host_memory_space> m = mat;
        tofile(fn,m);
    }

    template<class T>
    cuv::tensor<T, cuv::host_memory_space>
    fromfile(const std::string& fn){
        cuv::tensor<T, cuv::host_memory_space> ret;
        cnpy::NpyArray npymat = cnpy::npy_load(fn);
        ret.resize(npymat.shape);
        memcpy(ret.ptr(), npymat.data, ret.memsize());
        return ret;
    }
}
#endif
