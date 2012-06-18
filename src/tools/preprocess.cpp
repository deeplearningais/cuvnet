
#include <iostream>
#include <fstream>

void normalize_hog(cv::Mat& m, const unsigned int rows, const unsigned int cols, unsigned int steps){
    float* it    = m.ptr<float>();
    float* end   = it+rows*cols;
    for(; it<end; ++it){
        float sum=0;
        for (unsigned int s = 0; s < steps; ++s) sum += it[s*rows*cols];
        sum = 1.f/std::max(sum,0.0001f);
        for (unsigned int s = 0; s < steps; ++s) it[s*rows*cols]*=sum;
#if 1
        /* Hys2 norm */
        for (unsigned int s = 0; s < steps; ++s)
            it[s*rows*cols]=std::min(0.2f,it[s*rows*cols]);
        for (unsigned int s = 0; s < steps; ++s) sum += it[s*rows*cols];
        sum = 1.f/std::max(sum,0.0001f);
        for (unsigned int s = 0; s < steps; ++s) it[s*rows*cols]*=sum;
#endif
    }

}
cv::Mat dense_hog(const cv::Mat img, const int steps=5, const int spatialpool=3){
    cv::Mat gmx, gmy, mag, ang;
    cv::Scharr(img, gmx, CV_32F, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    cv::Scharr(img, gmy, CV_32F, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    //int Hshape[] = {steps,img.rows,img.cols};
    cv::Mat H(steps,img.rows*img.cols, CV_32F);
    H = cv::Scalar(0);

    mag.create(img.rows, img.cols, CV_32F);
    ang.create(img.rows, img.cols, CV_32F);

    cv::MatConstIterator_<cv::Vec3f> xit = gmx.begin<cv::Vec3f>(),
                             xit_end     = gmx.end<cv::Vec3f>(),
                             yit         = gmy.begin<cv::Vec3f>();
    cv::MatIterator_<float> mit = mag.begin<float>(),
                            ait = ang.begin<float>();

    size_t image_size = img.rows*img.cols;
    float div = 180./steps;
    //size_t cnt=0;
    float* Hptr = H.ptr<float>();

    for( ; xit != xit_end; ++xit, ++yit, ++mit, ++ait, ++Hptr )
    {
        const cv::Vec3f& X = *xit;
        const cv::Vec3f& Y = *yit;
        float maxval       = X[0]*X[0]+Y[0]*Y[0];
        int   arg_maxval   = 0;
        float f = X[1]*X[1]+Y[1]*Y[1];
        if(f>maxval){ maxval = f; arg_maxval=1; }
        f       = X[2]*X[2]+Y[2]*Y[2];
        if(f>maxval){ maxval = f; arg_maxval=2; }
        *mit = sqrt(maxval);
        float angle = cv::fastAtan2(Y[arg_maxval], X[arg_maxval]);
        angle = angle>=180.f ? angle-180.f : angle;
        *ait  = angle;

        int s    = angle/div;
        float a1 = div*s;
        float a2 = div*(s+1);
        Hptr[image_size*  s          ] += (a2-angle)/(a2-a1);
        Hptr[image_size*((s+1)%steps)] += (angle-a1)/(a2-a1);
    }

    for (int i = 0; i < steps; ++i)
    {
        cv::Mat slice = H.row(i).reshape(1, img.rows), slice2;
        //cv::boxFilter(slice,slice,CV_32F,cv::Size(spatialpool,spatialpool));
        cv::GaussianBlur(slice,slice,cv::Size(spatialpool,spatialpool), spatialpool/2.);
    }
    normalize_hog(H, img.rows, img.cols, steps);
    return H;
}

cv::Mat square(const cv::Mat& orig_, unsigned int sq_size){
    //unsigned int sq_size = std::max(orig.rows, orig.cols);
    unsigned int new_rows = orig_.rows >= orig_.cols ? sq_size : orig_.rows/(float)orig_.cols * sq_size;
    unsigned int new_cols = orig_.cols >= orig_.rows ? sq_size : orig_.cols/(float)orig_.rows * sq_size;
    cv::Mat orig;
    cv::resize(orig_,orig,cv::Size(new_cols, new_rows));
    unsigned int top=0, left=0;
    cv::Mat squared(sq_size, sq_size, orig.type(), cv::mean(orig));

    if(orig.rows!=orig.cols){
        if(orig.rows>orig.cols){
            left  += ceil((orig.rows-orig.cols)/2.);
        } else{
            top  += ceil ((orig.cols-orig.rows)/2.);
        }
    } 
    cv::Rect rect(left, top, orig.cols, orig.rows);

    cv::Mat roi(squared,rect);
    orig.copyTo(roi);
    for (int i = 0; i < 0; ++i)
    {
        cv::GaussianBlur(squared,squared,cv::Size(5,5),2.5,2.5);
        orig.copyTo(roi);
    }

    return squared;
}

using namespace cuvnet;

void patch_extractor::process_filestring(cuv::tensor<float,cuv::host_memory_space>& dst, const char* buf, size_t n){
    static const int s = 176;
    static const unsigned int n_hog_channels = 5;
    static const unsigned int hog_pool       = 3;
    cv::Mat orig, squared, H;
    cv::Mat buf_wrapper(n,1,CV_8UC1,const_cast<char*>(buf),1);
    orig    = cv::imdecode(buf_wrapper,1);
    squared = square(orig,s);
    H       = dense_hog(squared,n_hog_channels,hog_pool);
    dst.resize(cuv::extents[n_hog_channels][s][s]);
    cuv::tensor<float,cuv::host_memory_space> Hcuv(cuv::indices[cuv::index_range(0,n_hog_channels)][cuv::index_range(0,s)][cuv::index_range(0,s)], H.ptr<float>());
    dst = Hcuv;
}
