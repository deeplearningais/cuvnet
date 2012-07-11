#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/monitor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

#include <datasets/random_translation.hpp>
#include "cuvnet/models/relational_auto_encoder.hpp"
#include <vector>
using namespace cuvnet;
namespace ll = boost::lambda;
using namespace std;
/**
 * collects weights, inputs and decoded data from the network and plots it, and
 * writes the results to a file.
 *
 */

typedef boost::shared_ptr<ParameterInput> input_ptr;
typedef cuv::tensor<float,cuv::host_memory_space> tensor;



/// convenient transpose for a matrix (used in visualization only)
matrix trans(matrix& m){
    matrix mt(m.shape(1),m.shape(0));
    cuv::transpose(mt,m);
    return mt;
}

/**
 * enlarge a CxHxW channel in W and H dimension by an integer factor
 *
 * @param img the image to enlarge
 * @param zf  the zoom factor
 */
cuv::tensor<float,cuv::host_memory_space>
zoom(const cuv::tensor<float,cuv::host_memory_space>& img, int zf=16){
    cuv::tensor<float,cuv::host_memory_space> zimg(cuv::extents[img.shape(0)*zf][img.shape(1)*zf]);
    for(unsigned int i=0;i<zimg.shape(0); i++){
        for (unsigned int j = 0; j < zimg.shape(1); ++j)
        {
            zimg(i, j) = -img(i/zf,j/zf);
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
 * @param fs          input size
 * @param channels    number of channels of an image (should have shape channels X fs X fs)
 *
 * @return rearranged view
 *
 */
cuv::tensor<float,cuv::host_memory_space>
arrange_input_filters(const tensor& input_x_, const tensor& input_y_, const tensor& recon_,  unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, bool normalize_separately=false){
    tensor fx = input_x_.copy();
    tensor fy = input_y_.copy();
    tensor recon = recon_.copy();
    // normalize the values of the tensor between 0..1
    float min_ = min(min(cuv::minimum(fx), cuv::minimum(fx)), cuv::minimum(recon));
    float max_ = max(max(cuv::maximum(fx), cuv::maximum(fy)), cuv::maximum(recon));
    
    // ensures that we don't devide by zero
    if(max_ <  0.00001f)
       max_ = 0.00001f;

    fx -= min_; 
    fx /= max_;
    fy -= min_;
    fy /= max_;
    recon -= min_;
    recon /= max_;
    std::cout << " visualizing inputs and reconstruction " << std::endl;
    cuvAssert(!cuv::has_nan(fx));
    cuvAssert(!cuv::has_inf(fx));
    cuvAssert(!cuv::has_nan(fy));
    cuvAssert(!cuv::has_nan(fy));
    cuvAssert(!cuv::has_inf(recon));
    cuvAssert(!cuv::has_inf(recon));
    assert(fx.shape() == fy.shape() && fy.shape() == recon.shape());

    // the image which is visualized
    tensor img(cuv::extents[srcMapCount*4][dstMapCount*(fs + 1)]);
    img = 0.f;

    for(unsigned int sm=0; sm<srcMapCount; sm++){
        for (unsigned int dm = 0; dm < dstMapCount; ++dm) {
            int img_0 = sm * 4;
            int img_1 = dm*(fs+1);
            tensor f(cuv::extents[3][fs]);
            for(unsigned int elem=0;elem<fs;elem++){
                // first are all filters of 1st src map
                f(0, elem) = fx(sm*dstMapCount+dm, elem);
                f(1, elem) = fy(sm*dstMapCount+dm, elem);
                f(2, elem) = recon(sm*dstMapCount+dm, elem);
            }
            if(normalize_separately){
                 f -= cuv::minimum(f);
                 float max_ = cuv::maximum(f);
                 // ensures that we don't devide by zero
                 if(max_ <  0.00001f)
                    max_ = 0.00001f;
                 f /= max_;
            }
            for(unsigned int elem=0;elem<fs;elem++){
                img(img_0, img_1 + elem) = f(0,elem) ;
                img(img_0 + 1, img_1 + elem) = f(1,elem) ;
                img(img_0 + 2, img_1 + elem) = f(2,elem) ;
            }
        }
    }
    img = zoom(img);
    return img;
}


/**
 * arrange images stored in rows/columns of a matrix nicely for viewing
 *
 * @param w_          the matrix containing the images
 * @param transpose   if 't', transpose matrix before viewing rows
 * @param dstMapCount number of columns in the arrangement
 * @param srcMapCount number of rows in the arrangement
 * @param fs          input size
 * @param channels    number of channels of an image (should have shape channels X fs X fs)
 *
 * @return rearranged view
 *
 */
cuv::tensor<float,cuv::host_memory_space>
arrange_filters(const tensor& fx_, const tensor& fy_,  unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, bool normalize_separately=false){
    tensor fx = fx_.copy();
    tensor fy = fy_.copy();
    // normalize the values of the tensor between 0..1
    float max_ = max(cuv::maximum(fx), cuv::maximum(fy));
    float min_ = min(cuv::minimum(fx), cuv::minimum(fx));
    // ensures that we don't devide by zero
    //if(max_ <  0.00001f)
    //    max_ = 0.00001f;

    fx -= min_;
    fy -= min_;
    fx /= max_;
    fy /= max_;


    assert(fx.shape(0) == fy.shape(0) && fy.shape(1) == fy.shape(1));

    // the image which is visualized
    tensor img(cuv::extents[srcMapCount*3][dstMapCount*(fs + 1)]);
    img = 0.f;
    for(unsigned int sm=0; sm<srcMapCount; sm++){
        for (unsigned int dm = 0; dm < dstMapCount; ++dm) {
            int img_0 = sm * 3;
            int img_1 = dm*(fs+1);
            cuv::tensor<float,cuv::host_memory_space> f(cuv::extents[2][fs]);
            for(unsigned int elem=0;elem<fs;elem++){
                // first are all filters of 1st src map
                f(0, elem) = fx(sm*dstMapCount+dm, elem);
                f(1, elem) = fy(sm*dstMapCount+dm, elem);
            }
            if(normalize_separately){
                 f -= cuv::minimum(f);
                 float max_ = cuv::maximum(f);
                 // ensures that we don't devide by zero
                 //if(max_ <  0.00001f)
                 //    max_ = 0.00001f;
                 f /= max_;
            }
            for(unsigned int elem=0;elem<fs;elem++){
                img(img_0, img_1 + elem) = f(0,elem) ;
                img(img_0 + 1, img_1 + elem) = f(1,elem) ;
            }
        }
    }
    img = zoom(img);
    return img;
}

void visualize_filters(relational_auto_encoder* ae, monitor* mon, int fa,int fb, input_ptr input_x, input_ptr input_y, unsigned int epoch){
    if(epoch%300 != 0)
        return;
    {
        std::string base = (boost::format("fx&y-%06d-")%epoch).str();

        tensor fx = trans(ae->get_fx()->data());
        //fx *= fx;
        std::cout << "fx dims: "<<fx.shape(0)<<", "<<fx.shape(1)<<std::endl;

        tensor fy = trans(ae->get_fy()->data());
        std::cout << "fy dims: "<<fy.shape(0)<<", "<<fy.shape(1)<<std::endl;
        
        std::cout << " min elem fx: " << cuv::minimum(fx) << endl;
        auto wvis = arrange_filters(fx, fy, fa, fb, fx.shape(1),false);
        cuv::libs::cimg::save(wvis, base+"nb.png");

        wvis      = arrange_filters(fx, fy, fa, fb, fx.shape(1),true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
    }
    {
        std::string base = (boost::format("inputs-%06d-")%epoch).str();
        tensor in_x = input_x->data().copy();
        tensor in_y = input_y->data().copy();
        tensor recon = (*mon)["decoded"].copy();

        fa = sqrt(in_x.shape(0));
        fb = sqrt(in_x.shape(0));
    cuvAssert(!cuv::has_nan(in_x));
    cuvAssert(!cuv::has_inf(in_x));
    std::cout << "min max x: "<<cuv::minimum(in_x) << ", "<<cuv::maximum(in_y)<<std::endl;;
    std::cout << "min max y: "<<cuv::minimum(in_y) << ", "<<cuv::maximum(in_y)<<std::endl;;
        auto wvis = arrange_input_filters(in_x, in_y, recon, fa, fb, (int)in_x.shape(1),false);
        cuv::libs::cimg::save(wvis, base+"nb.png");
        wvis      = arrange_input_filters(in_x, in_y, recon, fa, fb, (int)in_x.shape(1),true);
        cuv::libs::cimg::save(wvis, base+"sb.png");
    }

}

/**
 * load a batch from the dataset
 */
void load_batch(
        boost::shared_ptr<ParameterInput> input_x,
        boost::shared_ptr<ParameterInput> input_y,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        unsigned int bs, unsigned int batch){
    input_x->data() = (*data)[cuv::indices[0][cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    input_y->data() = (*data)[cuv::indices[1][cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    cuvAssert(!cuv::has_nan(input_x->data()));
    cuvAssert(!cuv::has_inf(input_x->data()));
}


int main(int argc, char **argv)
{
    // initialize cuv library
    cuv::initCUDA(2);
    cuv::initialize_mersenne_twister_seeds();
    unsigned int fb=20,bs= 640, subsampling = 3, max_trans = 9, gauss_dist = 2;
    unsigned int fa = max_trans * 2 + 1;
    float sigma = 2.f, learning_rate = 0.3f;
    // generate random translation dataset
    std::cout << "generating dataset: "<<std::endl;
    //random_translation ds(100, 20, 10, 0.5f, 3, 2.f, 5, 10);
    //random_translation ds(20, 64* 200, 1024, 0.1f, 8, 5.f, 2, 3);
    random_translation ds(fb * subsampling, 10 * 64, 1024, 0.1f, gauss_dist, sigma, subsampling, max_trans);
    ds.binary   = false;

    // number of filters is fa*fb (fa and fb determine layout of plots printed
    //          in \c visualize_filters)
    // batch size is bs
    //unsigned int fa=16,fb=8,bs=512;
    //unsigned int fa=5,fb=20,bs= 4;

    std::cout << "Traindata: "<<std::endl;
    std::cout << ds.train_data.shape(0)<<std::endl;
    std::cout << ds.train_data.shape(1)<<std::endl;
    std::cout << ds.train_data.shape(2)<<std::endl;
    

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input correctly
    // \c load_batch will put values in it.
    input_ptr input_x(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(2)],"input_x")); 

    input_ptr input_y(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(2)],"input_y")); 

    relational_auto_encoder ae(fa, fa*fb, ds.binary); // creates simple autoencoder
    ae.init(input_x, input_y);
    

    // obtain the parameters which we need to derive for in the unsupervised
    // learning phase
    std::vector<Op*> params = ae.unsupervised_params();

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, ae.loss(),        "total loss");
    mon.add(monitor::WP_SINK,               ae.get_decoded_y(), "decoded");

    // copy training data to the device
    matrix train_data = ds.train_data;

    // create a \c gradient_descent object that derives the auto-encoder loss
    // w.r.t. \c params and has learning rate 0.001f
    //gradient_descent gd(ae.loss(),0,params, learning_rate);
    rprop_gradient_descent gd(ae.loss(), 0, params, 0.0000001);
    
    // register the monitor so that it receives learning events
    gd.register_monitor(mon);

    // after each epoch, run \c visualize_filters
    gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&mon,fa,fb, input_x, input_y,_1));

    // before each batch, load data into \c input
    gd.before_batch.connect(boost::bind(load_batch,input_x, input_y,&train_data,bs,_2));

    // the number of batches is constant in our case (but has to be supplied as a function)
    gd.current_batch_num.connect(ds.train_data.shape(1)/ll::constant(bs));

    // do mini-batch learning for at most 6000 epochs, or 10 minutes
    // (whatever comes first)
    //gd.minibatch_learning(50005, 100*60); // 10 minutes maximum
    load_batch(input_x, input_y, &train_data,bs,0);
    gd.batch_learning(10000, 100*60);
    
    return 0;
}
