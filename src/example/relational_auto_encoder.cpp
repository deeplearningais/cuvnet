#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>

#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/monitor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

#include <datasets/random_translation.hpp>
#include "cuvnet/models/relational_auto_encoder.hpp"
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/simple_auto_encoder.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/linear_regression.hpp>
#include <vector>
#include <map>
#include <exception>


using namespace cuvnet;
namespace ll = boost::lambda;
using namespace std;
/**
 * collects weights, inputs and decoded data from the network and plots it, and
 * writes the results to a file.
 *
 */

typedef boost::shared_ptr<ParameterInput> input_ptr;
typedef cuv::tensor<float,cuv::host_memory_space> tensor_type;
typedef boost::shared_ptr<Op>     op_ptr;


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
tensor_type
zoom(const tensor_type& img, int zf=16){
    tensor_type zimg(cuv::extents[img.shape(0)*zf][img.shape(1)*zf]);
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
 * arrange inputs stored in rows/columns of a matrix nicely for viewing
 *
 * @param input_x_             the matrix containing the first inputs
 * @param input_y_             the matrix containing the second inputs
 * @param recon_               the matrix containing the reconstruction of second input
 * @param pred_                the matrix containing the prediction 
 * @param dstMapCount          number of columns in the arrangement
 * @param srcMapCount          number of rows in the arrangement
 * @param fs                   input size
 * @param normalize_separately if true, normalizes each example separately  
 *
 * @return rearranged view
 *
 */
tensor_type
arrange_inputs_and_prediction(const tensor_type& input_x_, const tensor_type& input_y_, const tensor_type& recon_, const tensor_type& pred_, unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, bool normalize_separately=false){
    tensor_type fx = input_x_.copy();
    tensor_type fy = input_y_.copy();
    tensor_type recon = recon_.copy();
    tensor_type pred = pred_.copy();
    // normalize the values of the tensor between 0..1
    float min_ = min(min(min(cuv::minimum(fx), cuv::minimum(fx)), cuv::minimum(recon)), cuv::minimum(pred));
    float max_ = max(max(max(cuv::maximum(fx), cuv::maximum(fy)), cuv::maximum(recon)), cuv::maximum(pred));
    
    // ensures that we don't devide by zero
    if(max_ <  0.00001f)
       max_ = 0.00001f;

    fx -= min_; 
    fx /= max_;
    fy -= min_;
    fy /= max_;
    recon -= min_;
    recon /= max_;
    pred -= min_;
    pred /= max_;
    cuvAssert(!cuv::has_nan(fx));
    cuvAssert(!cuv::has_inf(fx));
    cuvAssert(!cuv::has_nan(fy));
    cuvAssert(!cuv::has_nan(fy));
    cuvAssert(!cuv::has_inf(recon));
    cuvAssert(!cuv::has_inf(recon));
    cuvAssert(!cuv::has_inf(pred));
    cuvAssert(!cuv::has_inf(pred));
    assert(fx.shape() == fy.shape() && fy.shape() == recon.shape() && recon.shape() == pred.shape());

    // the image which is visualized
    tensor_type img(cuv::extents[srcMapCount*5][dstMapCount*(fs + 1)]);
    img = 0.5f;

    for(unsigned int sm=0; sm<srcMapCount; sm++){
        for (unsigned int dm = 0; dm < dstMapCount; ++dm) {
            int img_0 = sm * 5;
            int img_1 = dm*(fs+1);
            tensor_type f(cuv::extents[4][fs]);
            for(unsigned int elem=0;elem<fs;elem++){
                // first are all filters of 1st src map
                f(0, elem) = fx(sm*dstMapCount+dm, elem);
                f(1, elem) = fy(sm*dstMapCount+dm, elem);
                f(2, elem) = recon(sm*dstMapCount+dm, elem);
                f(3, elem) = pred(sm*dstMapCount+dm, elem);
                
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
                img(img_0 + 3, img_1 + elem) = f(3,elem) ;
            }
        }
    }
    img = zoom(img);
    return img;
}

/**
 * arrange inputs stored in rows/columns of a matrix nicely for viewing
 *
 * @param input_x_          the matrix containing the first inputs
 * @param input_y_          the matrix containing the second inputs
 * @param recon_          the matrix containing the reconstruction of second input
 * @param dstMapCount number of columns in the arrangement
 * @param srcMapCount number of rows in the arrangement
 * @param fs          input size
 * @param normalize_separately if true, normalizes each example separately  
 *
 * @return rearranged view
 *
 */
tensor_type
arrange_input_filters(const tensor_type& input_x_, const tensor_type& input_y_, const tensor_type& recon_,  unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, bool normalize_separately=false){
    tensor_type fx = input_x_.copy();
    tensor_type fy = input_y_.copy();
    tensor_type recon = recon_.copy();
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
    cuvAssert(!cuv::has_nan(fx));
    cuvAssert(!cuv::has_inf(fx));
    cuvAssert(!cuv::has_nan(fy));
    cuvAssert(!cuv::has_nan(fy));
    cuvAssert(!cuv::has_inf(recon));
    cuvAssert(!cuv::has_inf(recon));
    assert(fx.shape() == fy.shape() && fy.shape() == recon.shape());

    // the image which is visualized
    tensor_type img(cuv::extents[srcMapCount*4][dstMapCount*(fs + 1)]);
    img = 0.f;

    for(unsigned int sm=0; sm<srcMapCount; sm++){
        for (unsigned int dm = 0; dm < dstMapCount; ++dm) {
            int img_0 = sm * 4;
            int img_1 = dm*(fs+1);
            tensor_type f(cuv::extents[3][fs]);
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
 * @param fx_          the matrix containing the filters Fx
 * @param fy_          the matrix containing the filters Fy
 * @param dstMapCount number of columns in the arrangement
 * @param srcMapCount number of rows in the arrangement
 * @param fs          input size
 * @param normalize_separately if true, normalizes each example separately  
 *
 * @return rearranged view
 *
 */
tensor_type
arrange_filters(const tensor_type& fx_, const tensor_type& fy_,  unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, bool normalize_separately=false){
    tensor_type fx = fx_.copy();
    tensor_type fy = fy_.copy();
    // normalize the values of the tensor between 0..1
    float max_ = max(cuv::maximum(fx), cuv::maximum(fy));
    float min_ = min(cuv::minimum(fx), cuv::minimum(fx));
    // ensures that we don't devide by zero
    if(max_ <  0.00001f)
        max_ = 0.00001f;

    fx -= min_;
    fy -= min_;
    fx /= max_;
    fy /= max_;


    assert(fx.shape(0) == fy.shape(0) && fy.shape(1) == fy.shape(1));

    // the image which is visualized
    tensor_type img(cuv::extents[srcMapCount*3][dstMapCount*(fs + 1)]);
    img = 0.f;
    for(unsigned int sm=0; sm<srcMapCount; sm++){
        for (unsigned int dm = 0; dm < dstMapCount; ++dm) {
            int img_0 = sm * 3;
            int img_1 = dm*(fs+1);
            tensor_type f(cuv::extents[2][fs]);
            for(unsigned int elem=0;elem<fs;elem++){
                // first are all filters of 1st src map
                f(0, elem) = fx(sm*dstMapCount+dm, elem);
                f(1, elem) = fy(sm*dstMapCount+dm, elem);
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
            }
        }
    }
    img = zoom(img);
    return img;
}

/**
 * arrange images stored in rows/columns of a matrix nicely for viewing
 *
 * @param fx_          the matrix containing the filters Fx
 * @param fy_          the matrix containing the filters Fy
 * @param dstMapCount number of columns in the arrangement
 * @param srcMapCount number of rows in the arrangement
 * @param fs          input size
 * @param normalize_separately if true, normalizes each example separately  
 *
 * @return rearranged view
 *
 */
tensor_type
arrange_single_filter(const tensor_type& fx_ ,  unsigned int dstMapCount, unsigned int srcMapCount, unsigned int fs, bool normalize_separately=false){
    tensor_type fx = fx_.copy();

    // the image which is visualized
    tensor_type img(cuv::extents[srcMapCount * 2][dstMapCount*(fs + 1)]);
    img = 0.f;
    for(unsigned int sm=0; sm<srcMapCount; sm++){
        for (unsigned int dm = 0; dm < dstMapCount; ++dm) {
            int img_0 = sm * 2;
            int img_1 = dm*(fs+1);
            tensor_type f(cuv::extents[fs]);
            for(unsigned int elem=0;elem<fs;elem++){
                // first are all filters of 1st src map
                f(elem) = fx(sm*dstMapCount+dm, elem);
            }
            //if(normalize_separately){
            //     f -= cuv::minimum(f);
            //     float max_ = cuv::maximum(f);
            //     // ensures that we don't devide by zero
            //     if(max_ <  0.00001f)
            //        max_ = 0.00001f;
            //     f /= max_;
            //}
            for(unsigned int elem=0;elem<fs;elem++){
                img(img_0, img_1 + elem) = f(elem);
            }
        }
    }
    img = zoom(img);
    return img;
}


/**
 * visualizes inputs, hidden layer, factors, filters 
 *
 * @param ae          the relational auto-encoder
 * @param mon         monitor, which is storing the intermediete values of loss
 * @param fa          number of columns in the arrangement
 * @param fb          number of rows in the arrangement
 * @param input_x     first input of the relational auto-encoder
 * @param input_y     second input of the relational auto-encoder
 * @param teacher     teacher of the relational auto-encoder
 * @param is_train_set if true, sets the file name for the train set, otherwise for the test set  
 * @param epoch        the current number of epoch  
 *
 * @return rearranged view
 *
 */
void visualize_filters(relational_auto_encoder* ae, monitor* mon, int fa,int fb, input_ptr input_x, input_ptr input_y, input_ptr teacher, bool is_train_set, unsigned int epoch){
    //if((epoch%1000 != 0 || epoch <= 1) && is_train_set)
    if(epoch%300 != 0   && is_train_set)
        return;
     std::cout << " visualizing inputs " << std::endl;
    //{
    //std::string base = (boost::format("fx&y-%06d-")%epoch).str();

    //tensor_type fx = trans(ae->get_fx()->data());

    //tensor_type fy = trans(ae->get_fy()->data());
        
    //std::cout << " min elem fx: " << cuv::minimum(fx) << endl;
    //auto wvis = arrange_filters(fx, fy, fa, fb, fx.shape(1),false);
    //cuv::libs::cimg::save(wvis, base+"nb.png");

    //wvis      = arrange_filters(fx, fy, fa, fb, fx.shape(1),true);
    //cuv::libs::cimg::save(wvis, base+"sb.png");
    //}
    {
     std::string base;
     if(is_train_set)
        base = (boost::format("inputs-train-%06d-")%epoch).str();
     else
        base = (boost::format("inputs-test-%06d-")%epoch).str();
     tensor_type in_x = input_x->data().copy();
     tensor_type in_y = input_y->data().copy();
     tensor_type prediction = (*mon)["decoded"].copy();
     //apply_scalar_functor(prediction,cuv::SF_SIGM);
     tensor_type teacher_ = teacher->data().copy();

     fa = 8;
     //fb = in_x.shape(0) / 10;
     fb =  8;
     cuvAssert(!cuv::has_nan(in_x));
     cuvAssert(!cuv::has_inf(in_x));
     auto wvis      = arrange_inputs_and_prediction(in_x, in_y, teacher_, prediction, fa, fb, (int)in_x.shape(1),true);
     cuv::libs::cimg::save(wvis, base+"sb.png");
     // wvis = arrange_inputs_and_prediction(in_x, in_y, teacher_, prediction, fa, fb, (int)in_x.shape(1),false);
     //cuv::libs::cimg::save(wvis, base+"nb.png");
    }
    ////{
    ////  std::string base = (boost::format("factors-%06d-")%epoch).str();
    ////  tensor_type fx = (*mon)["factorx"].copy();
    ////  tensor_type fy = (*mon)["factory"].copy();
    ////  tensor_type elem_mult = fx * fy;
    ////  fa = 10;
    ////  fb = fx.shape(0) / 10;

    ////  auto wvis = arrange_input_filters(fx, fy, elem_mult, fa, fb, (int)fx.shape(1), false);
    ////  cuv::libs::cimg::save(wvis, base+".png");

    //}
    //{
    //std::string base = (boost::format("encoder-%06d-")%epoch).str();
    //tensor_type encoder = (*mon)["encoded"].copy();
    //fa = 8;
    //fb =  8;

    //auto wvis = arrange_single_filter(encoder, fa, fb, (int)encoder.shape(1), false);
    //cuv::libs::cimg::save(wvis, base+".png");

    //}

}

/**
 * load a batch from the dataset
 *
 * @param input_x     first input of the relational auto-encoder
 * @param input_y     second input of the relational auto-encoder
 * @param teacher     teacher of the relational auto-encoder
 * @param data        the dataset, from which the batches are loaded
 */
void load_batch(
        boost::shared_ptr<ParameterInput> input_x,
        boost::shared_ptr<ParameterInput> input_y,
        boost::shared_ptr<ParameterInput> teacher,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        unsigned int bs, unsigned int batch){
    input_x->data() = (*data)[cuv::indices[0][cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    input_y->data() = (*data)[cuv::indices[1][cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    teacher->data() = (*data)[cuv::indices[2][cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    cuvAssert(!cuv::has_nan(input_x->data()));
    cuvAssert(!cuv::has_inf(input_x->data()));
}


/**
 * calculates the loss on the test set, used in the early stopping of the gradient descent
 *
 * @param mon         monitor, which is storing the intermediete values of loss
 * @param orig_gd     gradient descent of the training phase, its swiper is repaiered at the end of the function 
 * @param ds          dataset
 * @param ae          relational auto encoder
 * @param input_x     first input of the relational auto-encoder
 * @param input_y     second input of the relational auto-encoder
 * @param teacher     teacher of the relational auto-encoder
 * @param bs          batch size
 *
 */
float test_phase_early_stopping(monitor* mon, gradient_descent* orig_gd, random_translation* ds, relational_auto_encoder* ae, input_ptr input_x, input_ptr input_y, input_ptr teacher, int bs){
    float mean = 0.f;
    {
        tensor_type fx = ae->get_fx()->data();
        tensor_type fy = ae->get_fy()->data();
        std::cout << " min fx = " << cuv::minimum(fx) << " min fy " << cuv::minimum(fy) << std::endl;

        matrix data = ds->test_data;
        std::vector<Op*> params; // empty!
        rprop_gradient_descent gd(ae->loss(), 0, params,   0);
        //gradient_descent gd(ae->loss(), 0, params, 0);
        gd.register_monitor(*mon);
        gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &data, bs,_2));
        gd.current_batch_num.connect(data.shape(1)/ll::constant(bs));
        std::cout << std::endl << " testing phase ";
        mon->set_training_phase(false);
        gd.minibatch_learning(1, 100, 0);
        //load_batch(input_x, input_y, teacher, &data,bs,0);
        //gd.batch_learning(1, 100*60);
        mon->set_training_phase(true);
        mean = mon->mean("total loss");
    }
    orig_gd->repair_swiper();
    return mean;
}


/**
 * during the prediction phase, replaces the first input with second, and second input with prediction
 *
 * @param mon         monitor, which is storing the intermediete values of loss
 * @param input_x     first input of the relational auto-encoder
 * @param input_y     second input of the relational auto-encoder
 * @param epoch       current epoch
 *
 */
void generate_data(monitor* mon, input_ptr input_x, input_ptr input_y, int epoch){
    if(epoch > 0){
        tensor_type prediction = (*mon)["decoded"].copy();
        //apply_scalar_functor(prediction,cuv::SF_SIGM);
        for(unsigned int i = 0; i < prediction.shape(1); i++){
            for(unsigned int ex = 0; ex < prediction.shape(0); ex++){
                input_x->data()(ex,i) = input_y->data()(ex,i);
                input_y->data()(ex,i) = prediction(ex,i);
            }
        }
        //input_x->data() = input_y->data().copy();
        //input_y->data() = prediction.copy();

    }
}


/**
 * arrange  predictions, hidden units, factorx and factory in one image
 *
 * @param img             the matrix containing the image where everything is visualized
 * @param recon_          the matrix containing the reconstruction of second input
 * @param encoder         the matrix containing the hidden units 
 * @param factor_x        the matrix containing the factor_x for each example in the batch 
 * @param factor_y        the matrix containing the factor_y for each example in the batch 
 * @param fs              input size
 * @param epoch           current epoch
 * @param file_name       file name of the image
 *
 * @return rearranged view
 *
 */
tensor_type
arrange_predictions(tensor_type& img, const tensor_type& recon, const tensor_type& encoder, const tensor_type& factor_x, const tensor_type& factor_y, unsigned int fs,  int epoch, std::string file_name, int num_examples, int num_epochs, int ex){
    int hidden_size = encoder.shape(1);
    int num_factors = factor_x.shape(1);

    tensor_type rec(cuv::extents[fs]); 
    for(unsigned int elem=0;elem<fs;elem++){
         rec(elem) = recon(ex,elem);
    }
    rec -= cuv::minimum(rec);
    rec /= cuv::maximum(rec);
    for(unsigned int elem=0;elem<fs;elem++){
        img(epoch + 2, elem) = rec(elem);
    }

    tensor_type enc(cuv::extents[hidden_size]); 
    for(int elem=0;elem< hidden_size;elem++){
        enc(elem) =encoder(ex,elem);
    }
    enc -= cuv::minimum(enc);
    enc /= cuv::maximum(enc);
    for(int elem=0;elem< hidden_size;elem++){
        img((num_epochs + 2) + epoch + 2, elem) =enc(elem);
    }

    tensor_type fx(cuv::extents[num_factors]); 
    for(int elem=0;elem< num_factors;elem++){
        fx(elem) = factor_x(ex,elem);
    }
    fx -= cuv::minimum(fx);
    fx /= cuv::maximum(fx);
    for(int elem=0;elem< num_factors;elem++){
        img((num_epochs + 2) * 2 + epoch + 2, elem) = fx(elem);
    }

    tensor_type fy(cuv::extents[num_factors]); 
    for(int elem=0;elem< num_factors;elem++){
        fy(elem) = factor_y(ex,elem);
    }
    fy -= cuv::minimum(fy);
    fy /= cuv::maximum(fy);
    for(int elem=0;elem< num_factors;elem++){
        img((num_epochs + 2) * 3 + epoch + 2, elem) = fy(elem);
    }


    tensor_type img_z = img.copy();
    img_z -= cuv::minimum(img_z);
    img_z /= cuv::maximum(img_z);

    img_z = zoom(img_z);

    cuv::libs::cimg::save(img_z, file_name+ (boost::format("-ex-%1%.png") % ex).str());
    return img;
}


/**
 * arrange  predictions, hidden units, factorx and factory in one image
 *
 * @param mon             the monitor where intermediate values of loss, predictions, factors are saved
 * @param img             the matrix containing the image where everything is visualized
 * @param input_x         the matrix containing the first input
 * @param input_y         the matrix containing the second input
 * @param teacher         the matrix containing the teacher
 * @param num_examples    how many examples are visiualized
 * @param ex              current example which is visiualized
 * @param epoch           current epoch
 *
 * @return rearranged view
 *
 */
void visualize_prediction(monitor* mon, vector<tensor_type>& img, input_ptr input_x, input_ptr input_y, input_ptr teacher, int num_examples, int num_epochs, int ex, int epoch){
        {
           std::cout << " visualizing predictions " << std::endl;
           std::string base;
           base = (boost::format("predictions-%06d-")%epoch).str();
            
           if (epoch == 0){
               tensor_type in_x = input_x->data().copy();
               tensor_type in_y = input_y->data().copy();
               for(int ex = 0; ex < num_examples; ex++){
                   for(unsigned int i = 0; i < in_x.shape(1); i++){
                       img[ex](0, i) = in_x(ex,i);
                       img[ex](1, i) = in_y(ex,i);
                   }
               }
           }
           tensor_type prediction = (*mon)["decoded"].copy();
           tensor_type encoder = (*mon)["encoded"].copy();
           tensor_type factor_x = (*mon)["factorx"].copy();
           tensor_type factor_y = (*mon)["factory"].copy();
           //apply_scalar_functor(prediction,cuv::SF_SIGM);

           for(int ex = 0; ex < num_examples; ex++){
               img[ex] = arrange_predictions(img[ex], prediction, encoder, factor_x, factor_y, (int)prediction.shape(1), epoch, base, num_examples, num_epochs,ex);
           }
        }
}


/**
 * does minibatch learning on the training set
 *
 * @param ds          dataset
 * @param mon         monitor, which is storing the intermediete values of loss
 * @param ae          relational auto encoder
 * @param input_x     first input of the relational auto-encoder
 * @param input_y     second input of the relational auto-encoder
 * @param teacher     teacher of the relational auto-encoder
 * @param bs          batch size
 * @param input_size  the size of the input
 * @param num_hidden  number of hidden units 
 * @param params      parameters which are used during the learning 
 */
tensor_type train_phase(random_translation& ds, monitor& mon, relational_auto_encoder& ae, input_ptr input_x,input_ptr input_y, input_ptr teacher, int bs, int input_size, int num_hidden, std::vector<Op*>& params, unsigned int max_num_epochs){
        // copy training data to the device
        matrix train_data = ds.train_data;

        // create a \c gradient_descent object that derives the auto-encoder loss
        // w.r.t. \c params and has learning rate 0.001f
        //gradient_descent gd(ae.loss(),0,params, learning_rate);
        //rprop_gradient_descent gd(ae.loss(), 0, params, 0.00001, 0.005f);
        rprop_gradient_descent gd(ae.loss(), 0, params,   0.00001);
        //gd.setup_convergence_stopping(boost::bind(&monitor::mean, &mon, "total loss"), 0.45f,350);
        gd.setup_early_stopping(boost::bind(test_phase_early_stopping, &mon, &gd, &ds,  &ae,  input_x,  input_y, teacher,bs), 100, 1.f, 2.f);
        // register the monitor so that it receives learning events
        gd.register_monitor(mon);

        // after each epoch, run \c visualize_filters
        //gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&mon, num_hidden, input_size, input_x, input_y, teacher, true,_1));

        // before each batch, load data into \c input
        gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &train_data,bs,_2));

        // the number of batches is constant in our case (but has to be supplied as a function)
        gd.current_batch_num.connect(ds.train_data.shape(1)/ll::constant(bs));

        // do mini-batch learning for at most 6000 epochs, or 10 minutes
        // (whatever comes first)
        std::cout << std::endl << " Training phase: " << std::endl;
        //gd.minibatch_learning(5000, 100*60); // 10 minutes maximum
        //load_batch(input_x, input_y, teacher, &train_data,bs,0);
        //gd.batch_learning(max_num_epochs, 100*60);
        gd.minibatch_learning(max_num_epochs, 100*60, 0);
        tensor_type encoder_train = mon[("encoded")];
        return encoder_train;
}

/**
 * does minibatch learning on the test set
 *
 * @param ds          dataset
 * @param mon         monitor, which is storing the intermediete values of loss
 * @param ae          relational auto encoder
 * @param input_x     first input of the relational auto-encoder
 * @param input_y     second input of the relational auto-encoder
 * @param teacher     teacher of the relational auto-encoder
 * @param bs          batch size
 * @param input_size  the size of the input
 * @param num_hidden  number of hidden units 
 */
tensor_type test_phase(random_translation& ds, monitor& mon, relational_auto_encoder& ae, input_ptr input_x,input_ptr input_y, input_ptr teacher, int bs, int input_size, int num_hidden){
    std::cout << std::endl << " Test phase: " << std::endl;
    //// evaluates test data. We use minibatch learning with learning rate zero and only one epoch.
    matrix data = ds.test_data;
    std::vector<Op*> params; // empty!
    rprop_gradient_descent gd(ae.loss(), 0, params, 0);
    //gradient_descent gd(ae.loss(),0,params,0.f);
    gd.register_monitor(mon);
    //gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&mon,num_hidden,input_size, input_x, input_y, teacher, false,_1));
    //gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &data, bs,_2));
    gd.current_batch_num.connect(data.shape(1)/ll::constant(bs));
    //gd.minibatch_learning(1, 100, 0);
    load_batch(input_x, input_y, teacher, &data,bs,0);
    gd.batch_learning(1, 10);
    tensor_type encoder_test = mon[("encoded")];
    return encoder_test;
}

/**
 * makes multiple predictions of inputs
 *
 * @param ds          dataset
 * @param mon         monitor, which is storing the intermediete values of loss
 * @param ae          relational auto encoder
 * @param input_x     first input of the relational auto-encoder
 * @param input_y     second input of the relational auto-encoder
 * @param teacher     teacher of the relational auto-encoder
 * @param bs          batch size
 * @param input_size  the size of the input
 * @param num_hidden  number of hidden units 
 */
void prediction_phase(random_translation& ds, monitor& mon, relational_auto_encoder& ae, input_ptr input_x,input_ptr input_y, input_ptr teacher, int bs, int input_size, int num_hidden, int num_factors){
    std::cout << std::endl << " Prediction phase: " << std::endl;
    // the image which is visualized
    int num_epochs = 3;
    int num_examples = 12;
    matrix train_data = ds.train_data;

    //initializing image with first two inputs, later predictions are added
    int img_width = max(num_factors, max(num_hidden, input_size)); 
    vector<tensor_type> images;
    for(int ex = 0; ex < num_examples; ex++){
        tensor_type my_img(cuv::extents[4 * (num_epochs + 1 + 2)][img_width]);
        my_img = 0.f;
        images.push_back(my_img);
    }

    std::vector<Op*> params; // empty!
    rprop_gradient_descent gd(ae.loss(), 0, params, 0);
    //gradient_descent gd(ae.loss(),0,params,0.f);
    gd.register_monitor(mon);
    gd.before_epoch.connect(boost::bind(generate_data,&mon,input_x, input_y,_1));
    gd.current_batch_num.connect(ds.test_data.shape(1)/ll::constant(bs));
    gd.after_epoch.connect(boost::bind(visualize_prediction, &mon, images, input_x, input_y, teacher, num_examples, num_epochs, num_examples,  _1));
    //gd.minibatch_learning(num_epochs, 100, 0);
    //load_batch(input_x, input_y, teacher, &train_data,bs,0);

    //for (unsigned int i = 0; i < input_x->data().shape(1);i++){
    //    //translation by two to the left
    //    if(i >=20 && i <= 25) 
    //        input_x->data()(0,i) = 1.f; 
    //    else
    //        input_x->data()(0,i) = 0.f; 

    //    if(i >= 18 && i <= 23) 
    //        input_y->data()(0,i) = 1.f; 
    //    else
    //        input_y->data()(0,i) = 0.f; 

    //    if(i >=40 && i <= 47) 
    //        input_x->data()(1,i) = 1.f; 
    //    else
    //        input_x->data()(1,i) = 0.f; 

    //    if(i >= 38 && i <= 45) 
    //        input_y->data()(1,i) = 1.f; 
    //    else
    //        input_y->data()(1,i) = 0.f; 

    //    if(i >=60 && i <= 69) 
    //        input_x->data()(2,i) = 1.f; 
    //    else
    //        input_x->data()(2,i) = 0.f; 

    //    if(i >= 58 && i <= 67) 
    //        input_y->data()(2,i) = 1.f; 
    //    else
    //        input_y->data()(2,i) = 0.f; 




    //    //translation by two to the right
    //    if(i >=20 && i <= 25) 
    //        input_x->data()(3,i) = 1.f; 
    //    else
    //        input_x->data()(3,i) = 0.f; 

    //    if(i >= 22 && i <= 27) 
    //        input_y->data()(3,i) = 1.f; 
    //    else
    //        input_y->data()(3,i) = 0.f; 

    //    if(i >=40 && i <= 48) 
    //        input_x->data()(4,i) = 1.f; 
    //    else
    //        input_x->data()(4,i) = 0.f; 

    //    if(i >= 42 && i <= 50) 
    //        input_y->data()(4,i) = 1.f; 
    //    else
    //        input_y->data()(4,i) = 0.f; 


    //    if(i >=60 && i <= 70) 
    //        input_x->data()(5,i) = 1.f; 
    //    else
    //        input_x->data()(5,i) = 0.f; 

    //    if(i >= 62 && i <= 72) 
    //        input_y->data()(5,i) = 1.f; 
    //    else
    //        input_y->data()(5,i) = 0.f; 




    //    //translation by 1 to the right
    //    if(i >=20 && i <= 25) 
    //        input_x->data()(6,i) = 1.f; 
    //    else
    //        input_x->data()(6,i) = 0.f; 

    //    if(i >= 21 && i <= 26) 
    //        input_y->data()(6,i) = 1.f; 
    //    else
    //        input_y->data()(6,i) = 0.f; 


    //    if(i >=40 && i <= 49) 
    //        input_x->data()(7,i) = 1.f; 
    //    else
    //        input_x->data()(7,i) = 0.f; 

    //    if(i >= 41 && i <= 50) 
    //        input_y->data()(7,i) = 1.f; 
    //    else
    //        input_y->data()(7,i) = 0.f; 


    //    if(i >=30 && i <= 37) 
    //        input_x->data()(8,i) = 1.f; 
    //    else
    //        input_x->data()(8,i) = 0.f; 

    //    if(i >= 31 && i <= 38) 
    //        input_y->data()(8,i) = 1.f; 
    //    else
    //        input_y->data()(8,i) = 0.f; 



    //    //translation by 1 to the left
    //    if(i >=20 && i <= 25) 
    //        input_x->data()(9,i) = 1.f; 
    //    else
    //        input_x->data()(9,i) = 0.f; 

    //    if(i >= 19 && i <= 24) 
    //        input_y->data()(9,i) = 1.f; 
    //    else
    //        input_y->data()(9,i) = 0.f; 

    //    if(i >=40 && i <= 48) 
    //        input_x->data()(10,i) = 1.f; 
    //    else
    //        input_x->data()(10,i) = 0.f; 

    //    if(i >= 39 && i <= 47) 
    //        input_y->data()(10,i) = 1.f; 
    //    else
    //        input_y->data()(10,i) = 0.f; 


    //    if(i >=60 && i <= 69) 
    //        input_x->data()(11,i) = 1.f; 
    //    else
    //        input_x->data()(11,i) = 0.f; 

    //    if(i >= 59 && i <= 68) 
    //        input_y->data()(11,i) = 1.f; 
    //    else
    //        input_y->data()(11,i) = 0.f; 

    //}
    gd.batch_learning(num_epochs, 100*60);
}


/**
 * load a batch from the dataset for the logistic regression
 */
void load_batch_logistic(
        boost::shared_ptr<ParameterInput> input,
        boost::shared_ptr<ParameterInput> target,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        cuv::tensor<float,cuv::dev_memory_space>* labels,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    input->data() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    target->data() = (*labels)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
}


void regression(random_translation& ds, tensor_type& encoder_train, tensor_type& encoder_test){
    assert(encoder_train.shape(0) % 10 == 0);
    unsigned int bs = encoder_train.shape(0) / 10;
    std::cout << " size enc " << encoder_train.shape(0) << " size bs " << bs << std::endl; 

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input and target correctly
    // \c load_batch will put values in it.
    boost::shared_ptr<ParameterInput> input(
            new ParameterInput(cuv::extents[bs][encoder_train.shape(1)],"input"));
    boost::shared_ptr<ParameterInput> target(
            new ParameterInput(cuv::extents[bs][ds.train_labels.shape(1)],"target"));

    //creates stacked autoencoder with one simple autoencoder. has fa*fb number of hidden units
    auto_encoder_stack ae_s(false);
    typedef l2reg_simple_auto_encoder ae_type;
    ae_s.add<ae_type>(16*8, ds.binary, 0.01f);
    ae_s.init(input);

    // creates the logistic regression on the top of the stacked autoencoder
    logistic_regression lr(ae_s.get_encoded(), target);

    // puts the supervised parameters of the stacked autoencoder and logistic regression parameters in one vector
    std::vector<Op*> params = ae_s.supervised_params();
    std::vector<Op*> params_log = lr.params();
    std::copy(params_log.begin(), params_log.end(), std::back_inserter(params));

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true); // verbose
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, lr.get_loss(),        "total loss");
    mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, lr.classification_error(),        "classification error");

    // copy training data and labels to the device, and converts train_labels from int to float
    matrix train_data = encoder_train;
    matrix train_labels(ds.train_labels);
    
    std::cout << std::endl << " Training phase of logistic regression: " << std::endl;
    {
        // create a \c gradient_descent object that derives the logistic loss
        // w.r.t. \c params and has learning rate 0.1f
        rprop_gradient_descent gd(lr.get_loss(), 0, params,   0.00001);
        
        // register the monitor so that it receives learning events
        gd.register_monitor(mon);
        
        // after each epoch, run \c visualize_filters
        //gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&normalizer,fa,fb,ds.image_size,ds.channels, input,_1));
        
        // before each batch, load data into \c input
        gd.before_batch.connect(boost::bind(load_batch_logistic,input, target,&train_data, &train_labels, bs,_2));
        
        // the number of batches is constant in our case (but has to be supplied as a function)
        gd.current_batch_num.connect(encoder_train.shape(0) / ll::constant(bs));
        
        // do mini-batch learning for at most 100 epochs, or 10 minutes
        // (whatever comes first)
        //gd.minibatch_learning(10000, 10*60); // 10 minutes maximum
        gd.minibatch_learning(1000, 100*60, 0);
    }
    std::cout << std::endl << " Test phase: " << std::endl;

    // evaluates test data, does it similary as with train data, minibatch_learing is running only one epoch
    {
        matrix train_data = encoder_test;
        matrix train_labels(ds.test_labels);
        gradient_descent gd(lr.get_loss(),0,params,0.f);
        gd.register_monitor(mon);
        gd.before_batch.connect(boost::bind(load_batch_logistic,input, target,&train_data, &train_labels, bs,_2));
        gd.current_batch_num.connect(encoder_test.shape(0)/ll::constant(bs));
        gd.minibatch_learning(1);
    }

    std::cout <<  std::endl;
}


void initialize_pos_invariance_set(tensor_type& inv_data, int input_size, int ex, int trans, int factor, int subsample, int distance, float sigma){
    input_size *= subsample;
    inv_data.resize(cuv::extents[3][input_size * ex][input_size]);
    inv_data = 0.f;
    morse_code morse(inv_data, factor); 
    for(int ch = 0; ch < ex; ch++){
        morse.write_char(ch, 0, ch, 0); 
        morse.write_char(ch, 1, ch, get_wrap_index(input_size,trans)); 
        morse.write_char(ch, 2, ch, get_wrap_index(input_size, 2*trans)); 
    }
    inv_data = morse.get_data();

    for(int e = ex; e < input_size * ex; e++){
        for(int pos = 0; pos < input_size; pos++){ 
            for(int j = 0; j < 3; j++){ 
                inv_data(j, e, pos) = inv_data(j, e - ex, get_wrap_index(input_size, pos - 1));
            }
        } 

    }
    
    if(subsample > 1){
       //creates gaussian filter
       cuv::tensor<float,cuv::host_memory_space> gauss;
       fill_gauss(gauss, distance, sigma);

       // convolves last dim of both train and test data with the gauss filter
       convolve_last_dim(inv_data, gauss);

       // subsamples each "subsample" element
       subsampling(inv_data, subsample);
    }
    normalize_data_set(inv_data); 


}

void initialize_pattern_invariance_set(tensor_type& inv_data, int num_examples, int input_size, int trans, int factor, int subsample, int distance, float sigma){
    input_size *= subsample;
    inv_data.resize(cuv::extents[3][num_examples][input_size]);
    inv_data = 0.f;
    morse_code morse(inv_data, factor); 
    for(int ch = 0; ch < num_examples; ch++){
        morse.write_char(ch, 0, ch, input_size / 2); 
        morse.write_char(ch, 1, ch, get_wrap_index(input_size, input_size / 2 + trans)); 
        morse.write_char(ch, 2, ch, get_wrap_index(input_size, input_size / 2 + 2*trans)); 
    }
    inv_data = morse.get_data();
    
    if(subsample > 1){
       //creates gaussian filter
       cuv::tensor<float,cuv::host_memory_space> gauss;
       fill_gauss(gauss, distance, sigma);

       // convolves last dim of both train and test data with the gauss filter
       convolve_last_dim(inv_data, gauss);

       // subsamples each "subsample" element
       subsampling(inv_data, subsample);
    }
    normalize_data_set(inv_data); 
}

/**
 * arrange   hidden units
 *
 * @param encoder         the matrix containing the hidden units 
 *
 * @return rearranged view
 *
 */
tensor_type
arrange_hidden_activations(const tensor_type& encoder){
    unsigned int hidden_size = encoder.shape(1);
    unsigned int num_examples = encoder.shape(0);
    tensor_type img(cuv::extents[num_examples][hidden_size]);
    img = 0.f;
    for(unsigned int col = 0; col < hidden_size; col++){
        for(unsigned int row = 0; row < num_examples; row++){
            img(row, col) = encoder(row, col);
        } 
    }


    img = zoom(img);
    return img;
}

void visualize_invariance_set(monitor* mon, input_ptr input_x, input_ptr input_y, input_ptr teacher, int example, int trans){
    std::cout << " visualizing inputs " << std::endl;
    std::string base;
    if(example > 0){
        base = (boost::format("inputs-inv_%1%_%2%")% example % trans).str();
    }else{
        base = (boost::format("inputs-inv_%1%") % trans).str();
    }
    tensor_type in_x = input_x->data().copy();
    tensor_type in_y = input_y->data().copy();
    tensor_type prediction = (*mon)["decoded"].copy();
    //apply_scalar_functor(prediction,cuv::SF_SIGM);
    tensor_type teacher_ = teacher->data().copy();

    int fa = 4;
    int fb =  4;
    cuvAssert(!cuv::has_nan(in_x));
    cuvAssert(!cuv::has_inf(in_x));
    auto wvis      = arrange_inputs_and_prediction(in_x, in_y, teacher_, prediction, fa, fb, (int)in_x.shape(1),true);
    cuv::libs::cimg::save(wvis, base+".png");
    
    std::cout << " visualizing hidden activities " << endl;
    tensor_type encoder = (*mon)["encoded"].copy();

    if(example > 0){
        base = (boost::format("hidden-inv-pos_%1%_%2%") % trans % example).str();
    }else{
        base = (boost::format("hidden_inv-pattern-invariance_%1%") % trans).str();
    }

    wvis      = arrange_hidden_activations(encoder);
    cuv::libs::cimg::save(wvis, base+".png");

    tensor_type squared_ecnoder(cuv::extents[encoder.shape(0)][encoder.shape(1)]);
    tensor_type sqr_mean_enc(cuv::extents[encoder.shape(1)]);
    tensor_type mean_sqr_enc(cuv::extents[encoder.shape(1)]);
    tensor_type var(cuv::extents[encoder.shape(1)]);

    // E(x^2)
    apply_scalar_functor(squared_ecnoder, encoder,cuv::SF_SQUARE);
    reduce_to_row(mean_sqr_enc, squared_ecnoder, cuv::RF_MEAN);
    // E(x)^2
    reduce_to_row(sqr_mean_enc, encoder, cuv::RF_MEAN);
    apply_scalar_functor(sqr_mean_enc,cuv::SF_SQUARE);
    var = mean_sqr_enc - sqr_mean_enc; 

    std::cout << " Variance of hidden units " << std::endl;
    for(unsigned int i= 0; i < encoder.shape(1); i++){
        std::cout << "hidden unit " << i << ": " << var(i) << std::endl;
    }

}

void measure_invariance_position(relational_auto_encoder& ae, tensor_type& test_data, monitor& mon, input_ptr input_x, input_ptr input_y, input_ptr teacher, int example, int trans, int factor, int subsample, int distance, float sigma){

    std::cout << std::endl << " Measuring position invariance: " << std::endl;
    tensor_type inv_data;
    initialize_pos_invariance_set(inv_data,  test_data.shape(2), example, trans, factor, subsample, distance, sigma);
    std::cout << " inv_data size " << inv_data.shape(2) <<  endl;

    matrix data = inv_data;
    int bs = 300;

    std::vector<Op*> params; // empty!
    rprop_gradient_descent gd(ae.loss(), 0, params, 0);
    gd.register_monitor(mon);
    gd.after_epoch.connect(boost::bind(visualize_invariance_set, &mon, input_x, input_y, teacher, example, trans));
    gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &data, bs,_2));
    gd.current_batch_num.connect(data.shape(1)/ll::constant(bs));
    gd.minibatch_learning(1, 100, 0);
    //load_batch(input_x, input_y, teacher, &data, bs,0);
    //gd.batch_learning(1, 100);
}

void measure_pattern_invariance(relational_auto_encoder& ae, tensor_type& test_data, monitor& mon, input_ptr input_x, input_ptr input_y, input_ptr teacher, int num_examples, int trans, int factor, int subsample, int distance, float sigma){
    std::cout << std::endl << " Measuring pattern invariance: " << std::endl;
    tensor_type inv_data;
    initialize_pattern_invariance_set(inv_data, num_examples, test_data.shape(2),  trans, factor, subsample, distance, sigma);
    std::cout << " inv_data size " << inv_data.shape(2) <<  endl;

    matrix data = inv_data;
    int bs = inv_data.shape(1);
    std::cout << " bs " << bs <<  endl;

    std::vector<Op*> params; // empty!
    rprop_gradient_descent gd(ae.loss(), 0, params, 0);
    gd.register_monitor(mon);
    gd.after_epoch.connect(boost::bind(visualize_invariance_set, &mon, input_x, input_y, teacher, -1,  trans));
    gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &data, bs,_2));
    gd.current_batch_num.connect(data.shape(1)/ll::constant(bs));
    gd.minibatch_learning(1, 100, 0);
    //load_batch(input_x, input_y, teacher, &data, bs,0);
    //gd.batch_learning(1, 100);
}

class max_example_reached_exception: public exception{
};

class analyse{
    private:
        /// file where we write the hidden activations and parameters
        std::ofstream m_logfile;

        /// if true the header needs to be written in the file
        bool need_header_log_file;
        
        /// the name of the file
        string m_file_name;

    public:
        analyse(std::string file_name):
        need_header_log_file(true),
        m_file_name(file_name)
        {
        }
        void analyse_invariance(op_ptr op, boost::function<map<string,int>()> data_gen, int num_data=INT_MAX){
            // open file
            m_logfile.open(m_file_name.c_str(), std::ios::out);
            try{
                for(int i=0;i<num_data; i++){
                    map<string, int> param = data_gen();
                    cuvnet::function f(op);
                    matrix hidden_act = f.evaluate();
                    log_to_file(hidden_act, param); // write to file (repeatedly)
                }
            }catch(max_example_reached_exception){
            }
            m_logfile.close();
        }

        void log_to_file(matrix& hidden_act, map<string, int>& param){
            assert(m_logfile.is_open());
            map<string,int>::iterator it;
            if(need_header_log_file){
                for ( it=param.begin() ; it != param.end(); it++ ){
                    m_logfile << it->first << ",";
                }
                for (unsigned int i = 0; i < hidden_act.shape(1); ++i)
                {
                    if(i == hidden_act.shape(1)){
                        m_logfile << "h" << i;
                    }else{
                        m_logfile << "h" << i << ",";
                    }
                }
                need_header_log_file = false;
                m_logfile << std::endl;
            }

            for ( it=param.begin() ; it != param.end(); it++ ){
                m_logfile << it->second << ",";
            }
            for (unsigned int i = 0; i < hidden_act.shape(1); ++i)
            {
                if(i == hidden_act.shape(1)){
                    m_logfile << hidden_act(0,i);
                }else{
                    m_logfile << hidden_act(0,i) << ",";
                }
            }

            m_logfile << std::endl;
        }
};

struct morse_pat_gen{
    private:
    input_ptr m_input_x, m_input_y;
    
    int m_min_trans, m_max_trans;
    int m_min_pos, m_max_pos;
    int m_min_ex_type, m_max_ex_type;
    int m_factor, m_input_size;
    int m_max_num_examples;
    int m_current_example;
    public:

    morse_pat_gen(input_ptr input_x, input_ptr input_y, int min_trans, int max_trans, int min_pos, int max_pos, int min_ex_type, int max_ex_type, int factor, int input_size, int max_num_examples):
        m_input_x(input_x),
        m_input_y(input_y),
        m_min_trans(min_trans),
        m_max_trans(max_trans),
        m_min_pos(min_pos),
        m_max_pos(max_pos),
        m_min_ex_type(min_ex_type),
        m_max_ex_type(max_ex_type),
        m_factor(factor),
        m_input_size(input_size),
        m_max_num_examples(max_num_examples),
        m_current_example(0)
    {}
    map<string,int> operator()(){
        if(m_current_example == m_max_num_examples){
            throw max_example_reached_exception();
        }
        m_current_example++;
        srand ( time(NULL) );
        int trans = (rand() % (m_max_trans - m_min_trans + 1)) + m_min_trans;
        int pos = (rand() % (m_max_pos - m_min_pos + 1)) + m_min_pos;
        int ex_type = (rand() % (m_max_ex_type - m_min_ex_type + 1)) + m_min_ex_type;

        tensor_type example;                              
        morse_code morse(example, m_factor);
        morse.write_char(ex_type, 0, 0, pos);
        morse.write_char(ex_type, 0, 0, get_wrap_index(m_input_size, pos + trans));
        example = morse.get_data();
        m_input_x->data() = example[cuv::indices[0][cuv::index_range()][cuv::index_range()]];
        m_input_y->data() = example[cuv::indices[1][cuv::index_range()][cuv::index_range()]];
        map<string,int> param_map;
        param_map["translation"] = trans;
        param_map["position"] = pos;
        param_map["input_type"] = ex_type;
        return param_map;
    }
};



int main(int argc, char **argv)
{
    // initialize cuv library
    cuv::initCUDA(2);
    cuv::initialize_mersenne_twister_seeds();
    unsigned int input_size=100,bs=  300 , subsampling = 2, max_trans = 2, gauss_dist = 6, min_width = 10, max_width = 30, max_growing = 0, flag = 2, morse_factor = 6;
    unsigned max_num_epochs = 200;

    //unsigned int fa = (max_growing * 2 + 1) * (max_trans * 2 + 1) ;
    unsigned int num_hidden = 10;

    unsigned int num_factors = 200;
    float sigma = gauss_dist / 3; 
    //float learning_rate = 0.001f;
    // generate random translation datas
    unsigned int data_set_size = 3000;
    std::cout << "generating dataset: "<<std::endl;
    random_translation ds(input_size * subsampling,  data_set_size, data_set_size, 0.05f, gauss_dist, sigma, subsampling, max_trans, max_growing, min_width, max_width, flag, morse_factor);
    ds.binary   = false;
    //ds.binary   = true;


    std::cout << "Traindata: "<<std::endl;
    std::cout << "Number of examples : " << ds.train_data.shape(1)<<std::endl;
    std::cout << "Input size: " << ds.train_data.shape(2)<<std::endl;
    

    // an \c Input is a function with 0 parameters and 1 output.
    // here we only need to specify the shape of the input correctly
    // \c load_batch will put values in it.
    input_ptr input_x(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(2)],"input_x")); 

    input_ptr input_y(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(2)],"input_y")); 
    
    input_ptr teacher(
            new ParameterInput(cuv::extents[bs][ds.train_data.shape(2)],"teacher")); 
    
    std::cout << "creating relational auto-encoder with " << num_hidden << " hidden units and " << num_factors << " number of factors." << std::endl;
    relational_auto_encoder ae(num_hidden, num_factors, ds.binary); // creates simple autoencoder
    ae.init(input_x, input_y, teacher);
    

    // obtain the parameters which we need to derive for in the unsupervised
    // learning phase
    std::vector<Op*> params = ae.unsupervised_params();

    // create a verbose monitor, so we can see progress 
    // and register the decoded activations, so they can be displayed
    // in \c visualize_filters
    monitor mon(true);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, ae.loss(),        "total loss");
    mon.add(monitor::WP_SINK,               ae.get_decoded_y(), "decoded");
    mon.add(monitor::WP_SINK,               ae.get_factor_x(), "factorx");
    mon.add(monitor::WP_SINK,               ae.get_factor_y(), "factory");
    mon.add(monitor::WP_SINK,               ae.get_encoded(), "encoded");

    // does minibatch learning on the train set
    tensor_type encoder_train = train_phase( ds,  mon,  ae,  input_x, input_y,  teacher,  bs, input_size,  num_hidden,  params, max_num_epochs);
    
    // evaluates the test set
    tensor_type encoder_test = test_phase( ds,  mon,  ae,  input_x, input_y,  teacher,  bs, input_size,  num_hidden);

    // makes multiple predictions and visualizes inputs, predictions, hidden units, and factors 
    //prediction_phase( ds,  mon,  ae,  input_x, input_y,  teacher,  bs, input_size,  num_hidden, num_factors);
    
    //regression(ds, encoder_train, encoder_test);


    analyse a("invariance_test.txt");
    
    morse_pat_gen m(input_x, input_y, -max_trans, max_trans,  0,  input_size, 0, 38, morse_factor, input_size, 100);
    a.analyse_invariance(ae.get_encoded(), m); 

    //boost::bind(morse_pat_gen());

    //for(int trans = - max_trans; trans <= (int)max_trans; trans++){
    //    measure_invariance_position(ae,  ds.test_data,  mon,  input_x,  input_y,  teacher, num_examples, trans, morse_factor, subsampling, gauss_dist, sigma);
    //}
    //for(int trans = - max_trans; trans <= (int)max_trans; trans++){
    //   measure_pattern_invariance(ae,  ds.test_data,  mon,  input_x,  input_y,  teacher, num_examples , trans, morse_factor, subsampling, gauss_dist, sigma);
    //}

    return 0;
}
