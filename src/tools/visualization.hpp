#ifndef __CUVNET_VISUALIZATION_HPP__
#     define __CUVNET_VISUALIZATION_HPP__

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

namespace cuvnet
{

    cuv::tensor<float,cuv::host_memory_space>
        zoom(const cuv::tensor<float,cuv::host_memory_space>& img, int zf=2){
            cuv::tensor<float,cuv::host_memory_space> zimg(cuv::extents[img.shape(0)][img.shape(1)*zf][img.shape(2)*zf]);
            for(unsigned int c=0;c<zimg.shape(0);c++){
                for(unsigned int i=0;i<zimg.shape(1); i++){
                    for (unsigned int j = 0; j < zimg.shape(2); ++j)
                    {
                        zimg(c,i, j) = -img(c,i/zf,j/zf);
                    }
                }
            }
            zimg -= cuv::minimum(zimg);
            zimg *= 255 / cuv::maximum(zimg);
            return zimg;
        }
    cuv::tensor<float,cuv::host_memory_space>
        arrange_filters(const cuv::tensor<float,cuv::host_memory_space>& w, unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, unsigned int channels=1){
            cuv::tensor<float,cuv::host_memory_space> img(cuv::extents[channels][srcMapCount*(fs+1)][dstMapCount*(fs+1)]);
            img = 0.f;
            for(unsigned int sm=0; sm<srcMapCount; sm++){
                for (unsigned int dm = 0; dm < dstMapCount; ++dm) {
                    int img_0 = sm*(fs+1);
                    int img_1 = dm*(fs+1);
                    cuv::tensor<float,cuv::host_memory_space> f(cuv::extents[channels][fs][fs]);
                    for(unsigned int c=0;c<channels;c++)
                        for(unsigned int fx=0;fx<fs;fx++){
                            for(unsigned int fy=0;fy<fs;fy++){
                                // first are all filters of 1st src map
                                f(c,fy, fx) = w(sm*dstMapCount+dm, c*fs*fs + fy*fs+fx);
                            }
                        }
                    //f -= cuv::minimum(f);
                    //f /= cuv::maximum(f);
                    for(unsigned int c=0;c<channels;c++)
                        for(unsigned int fx=0;fx<fs;fx++){
                            for(unsigned int fy=0;fy<fs;fy++){
                                img(c,img_0+fy, img_1+fx) = f(c,fy, fx) ;
                            }
                        }
                }
            }
            img = zoom(img);
            if(img.shape(0)==1)
                img.reshape(cuv::extents[img.shape(1)][img.shape(2)]);
            return img;
        }

}
#endif /* __CUVNET_VISUALIZATION_HPP__ */
