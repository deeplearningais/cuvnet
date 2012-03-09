#ifndef __CUVNET_VISUALIZATION_HPP__
#     define __CUVNET_VISUALIZATION_HPP__

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

namespace cuvnet
{

    /**
     * enlarge a CxHxW channel in W and H dimension by an integer factor
     *
     * @param img the image to enlarge
     * @param zf  the zoom factor
     */
    cuv::tensor<float,cuv::host_memory_space>
        zoom(const cuv::tensor<float,cuv::host_memory_space>& img, int zf=4){
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

    /**
     * arrange images stored in rows/columns of a matrix nicely for viewing
     *
     * @param w_          the matrix containing the images
     * @param transpose   if 't', transpose matrix before viewing rows
     * @param dstMapCount number of columns in the arrangement
     * @param srcMapCount number of rows in the arrangement
     * @param fs          width and height of an image
     * @param channels    number of channels of an image (should have shape channels X fs X fs)
     *
     * @return rearranged view
     *
     */
    template<class T>
    cuv::tensor<float,cuv::host_memory_space>
        arrange_filters(const T& w_, char transpose,  unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, unsigned int channels=1, bool normalize_separately=false){
            cuv::tensor<float, cuv::host_memory_space> w = w_.copy();

            if(transpose=='t'){
                cuv::tensor<float,cuv::host_memory_space> wt(cuv::extents[w.shape(1)][w.shape(0)]);
                cuv::transpose(wt,w);
                w = wt;
            }
            w -= cuv::minimum(w);
            w /= cuv::maximum(w);

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
                    if(normalize_separately){
                        f -= cuv::minimum(f);
                        f /= cuv::maximum(f);
                    }
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
