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
#include <tools/dumper.hpp>
#include <tools/dataset_dumper.hpp>


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
void visualize_filters(relational_auto_encoder* ae, monitor* mon,  input_ptr input_x, input_ptr input_y, input_ptr teacher,std::string param_name,  bool is_train_set, unsigned int epoch){
    int fa, fb = 8;
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
        base = (boost::format("inputs-train-%1%")% param_name).str();
     else
        base = (boost::format("inputs-test-%1%")% param_name ).str();
     //tensor_type in_x = input_x->data().copy();
     //tensor_type in_y = input_y->data().copy();
     
     tensor_type in_x = (*mon)["x"].copy();
     tensor_type in_y = (*mon)["y"].copy();
     tensor_type prediction = (*mon)["decoded"].copy();
     //apply_scalar_functor(prediction,cuv::SF_SIGM);
     //tensor_type teacher_ = teacher->data().copy();
     tensor_type teacher_ = (*mon)["teacher"].copy();

     fa = 10;
     //fb = in_x.shape(0) / 10;
     fb =  30;
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
        std::cout << " min fx = " << cuv::minimum(fx) << " min fy " << cuv::minimum(fy) << " max fx = " << cuv::maximum(fx) << " max fy " << cuv::maximum(fy) << " min fh " << cuv::minimum(ae->get_fh()->data()) << " max fh " << cuv::maximum(ae->get_fh()->data()) << std::endl;

        matrix data = ds->test_data;
        std::vector<Op*> params; // empty!
        rprop_gradient_descent gd(ae->loss(), 0, params,   0);
        //gradient_descent gd(ae->loss(), 0, params, 0);
        mon->register_gd(gd);
        gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &data, bs,_2));
        gd.current_batch_num = data.shape(1)/ll::constant(bs);
        std::cout << std::endl << " testing phase ";
        mon->set_training_phase(CM_TEST);
        gd.minibatch_learning(1, 100, 0,false);
        //load_batch(input_x, input_y, teacher, &data,bs,0);
        //gd.batch_learning(1, 100*60);
        mon->set_training_phase(CM_TRAIN);
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

void log_activations(monitor * mon, dataset_dumper * dum, random_translation * ds, int bs, bool is_train_set, int batch){
    tensor_type act = (*mon)["encoded"];
    tensor_type labels;
    if(is_train_set){
        labels = ds->train_labels[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    }else{
        labels = ds->test_labels[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    }
    dum->write_to_file(act, labels);
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
void train_phase(random_translation& ds, monitor& mon, relational_auto_encoder& ae, input_ptr input_x,input_ptr input_y, input_ptr teacher, int bs, int input_size, int num_hidden, std::vector<Op*>& params, unsigned int max_num_epochs){
        // copy training data to the device
        matrix train_data = ds.train_data;

        // create a \c gradient_descent object that derives the auto-encoder loss
        // w.r.t. \c params and has learning rate 0.001f
        //gradient_descent gd(ae.loss(),0,params, learning_rate);
        //rprop_gradient_descent gd(ae.loss(), 0, params, 0.00001, 0.0005f);
        rprop_gradient_descent gd(ae.loss(), 0, params,   0.00001);
        //gd.setup_convergence_stopping(boost::bind(&monitor::mean, &mon, "total loss"), 0.45f,350);

        early_stopper es(gd, boost::bind(test_phase_early_stopping, &mon, &gd, &ds,  &ae,  input_x,  input_y, teacher,bs), 1.f, 100, 2.f);

        // register the monitor so that it receives learning events
        mon.register_gd(gd, es);

        // after each epoch, run \c visualize_filters

        // before each batch, load data into \c input
        gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &train_data,bs,_2));

        // the number of batches is constant in our case (but has to be supplied as a function)
        gd.current_batch_num = ds.train_data.shape(1)/ll::constant(bs);

        // do mini-batch learning for at most 6000 epochs, or 10 minutes
        // (whatever comes first)
        std::cout << std::endl << " Training phase: " << std::endl;
        //gd.minibatch_learning(5000, 100*60); // 10 minutes maximum
        //load_batch(input_x, input_y, teacher, &train_data,bs,0);
        //gd.batch_learning(max_num_epochs, 100*60);
        gd.minibatch_learning(max_num_epochs, 100*60, 0, false);

        // register dumper
        dataset_dumper dum("train_data.dat", ds.train_data.shape(1) / bs);
        gd.after_batch.connect(boost::bind(log_activations, &mon, &dum, &ds,bs, true, _2));
        gd.minibatch_learning(1, 100*60,0, false);
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
 * @param param_name  this string is appended to image name of the input patterns, describing the transformations used
 */
void test_phase(random_translation& ds, monitor& mon, relational_auto_encoder& ae, input_ptr input_x,input_ptr input_y, input_ptr teacher, int bs, int input_size, int num_hidden, std::string param_name){
    dataset_dumper dum("test_data.dat", ds.train_data.shape(1) / bs);
    std::cout << std::endl << " Test phase: " << std::endl;
    //// evaluates test data. We use minibatch learning with learning rate zero and only one epoch.
    matrix data = ds.test_data;
    std::vector<Op*> params; // empty!
    rprop_gradient_descent gd(ae.loss(), 0, params, 0);
    //gradient_descent gd(ae.loss(),0,params,0.f);
    mon.register_gd(gd);
    gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&mon, input_x, input_y, teacher, param_name, false,_1));
    gd.before_batch.connect(boost::bind(load_batch,input_x, input_y, teacher, &data, bs,_2));
    gd.after_batch.connect(boost::bind(log_activations, &mon, &dum, &ds,bs, false, _2));
    gd.current_batch_num = data.shape(1)/ll::constant(bs);
    gd.minibatch_learning(1, 100, 0, false);
    dum.close();
    //load_batch(input_x, input_y, teacher, &data,bs,0);
    //gd.batch_learning(1, 10);
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
    mon.register_gd(gd);
    gd.before_epoch.connect(boost::bind(generate_data,&mon,input_x, input_y,_1));
    gd.current_batch_num = ds.test_data.shape(1)/ll::constant(bs);
    gd.after_epoch.connect(boost::bind(visualize_prediction, &mon, images, input_x, input_y, teacher, num_examples, num_epochs, num_examples,  _1));
    //gd.minibatch_learning(num_epochs, 100, 0);
    //load_batch(input_x, input_y, teacher, &train_data,bs,0);

    gd.batch_learning(num_epochs, 100*60);
}


/**
 * load a batch from the dataset for the logistic regression
 */
void load_batch_logistic(
        linear_regression* lr,
        boost::shared_ptr<ParameterInput> input,
        boost::shared_ptr<ParameterInput> target,
        cuv::tensor<float,cuv::dev_memory_space>* data,
        cuv::tensor<float,cuv::dev_memory_space>* labels,
        unsigned int bs, unsigned int batch){
    //std::cout <<"."<<std::flush;
    input->data() = (*data)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    target->data() = (*labels)[cuv::indices[cuv::index_range(batch*bs,(batch+1)*bs)][cuv::index_range()]];
    tensor_type w = lr->get_weights()->data();
    //std::cout << " w min " << cuv::minimum(w) << " w max " << cuv::maximum(w) << std::endl; 
}


void regression(random_translation& ds, int bs){
    dataset_reader dum("train_data.dat", "test_data.dat");
    dum.read_from_file();

    tensor_type encoder_train = dum.train_data;
    tensor_type encoder_test = dum.test_data;

    tensor_type labels_train = dum.train_labels;
    tensor_type labels_test = dum.test_labels;

    for (int i = 0; i < encoder_train.shape(0); ++i)
    {
        for (int j = 0; j < encoder_train.shape(1); ++j)
        {
            std::cout << "ecoder tr " << labels_train(i,j) << "  ";
        }
    }


   // an \c Input is a function with 0 parameters and 1 output.
   // here we only need to specify the shape of the input and target correctly
   // \c load_batch will put values in it.
   boost::shared_ptr<ParameterInput> input(
           new ParameterInput(cuv::extents[bs][encoder_train.shape(1)],"input"));
   boost::shared_ptr<ParameterInput> target(
           new ParameterInput(cuv::extents[bs][ds.train_labels.shape(1)],"target"));





   // creates the linear regression 
   linear_regression lr( input, target);
   //linear_lasso_regression lr(0.1, input, target);
   //linear_ridge_regression lr(0.1, input, target);

   // puts the supervised parameters of the stacked autoencoder and logistic regression parameters in one vector
   std::vector<Op*> params = lr.params();

   // create a verbose monitor, so we can see progress 
   // and register the decoded activations, so they can be displayed
   // in \c visualize_filters
   monitor mon(true); // verbose
   mon.add(monitor::WP_SCALAR_EPOCH_STATS, lr.get_loss(),        "total loss");
   mon.add(monitor::WP_FUNC_SCALAR_EPOCH_STATS, lr.classification_error(),        "classification error");

    
   std::cout << std::endl << " Training phase of regression: " << std::endl;
   {
       // copy training data and labels to the device, and converts train_labels from int to float
       matrix train_data = encoder_train;
       matrix train_labels(labels_train);

      // create a \c gradient_descent object that derives the logistic loss
       // w.r.t. \c params and has learning rate 0.1f
       rprop_gradient_descent gd(lr.get_loss(), 0, params,   0.00001);
        
       // register the monitor so that it receives learning events
       mon.register_gd(gd);
        
       // after each epoch, run \c visualize_filters
       //gd.after_epoch.connect(boost::bind(visualize_filters,&ae,&normalizer,ds.image_size,ds.channels, input,_1));
        
       // before each batch, load data into \c input
       gd.before_batch.connect(boost::bind(load_batch_logistic, &lr, input, target,&train_data, &train_labels, bs,_2));
        
       // the number of batches is constant in our case (but has to be supplied as a function)
       gd.current_batch_num = encoder_train.shape(0) / ll::constant(bs);
        
       // do mini-batch learning for at most 100 epochs, or 10 minutes
       // (whatever comes first)
       //gd.minibatch_learning(10000, 10*60); // 10 minutes maximum
       gd.minibatch_learning(2000, 100*60, 0);
   }
   std::cout << std::endl << " Test phase: of regression " << std::endl;

   //evaluates test data, does it similary as with train data, minibatch_learing is running only one epoch
   {
      matrix test_data = encoder_test;
      matrix test_labels(labels_test);
      rprop_gradient_descent gd(lr.get_loss(), 0, params,   0.00001);
      mon.register_gd(gd);
      gd.before_batch.connect(boost::bind(load_batch_logistic, &lr, input, target,&test_data, &test_labels, bs,_2));
      gd.current_batch_num = encoder_test.shape(0)/ll::constant(bs);
      gd.minibatch_learning(1);
   }
   std::cout <<  std::endl;
}





struct morse_pat_gen{
    private:
    input_ptr m_input_x, m_input_y;
    
    int m_min_trans, m_max_trans;
    int m_min_pos, m_max_pos;
    int m_min_ex_type, m_max_ex_type;
    int m_factor, m_input_size;
    int m_max_num_examples;
    float m_max_scale;
    int m_current_example;
    int m_current_pos;
    float m_current_trans;
    float m_current_scale;
    int m_current_ex_type;
    public:

    morse_pat_gen(input_ptr input_x, input_ptr input_y, int min_trans, int max_trans, int min_pos, int max_pos, int min_ex_type, int max_ex_type, int factor, int input_size, int max_num_examples, float max_scale):
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
        m_max_scale(max_scale),
        m_current_example(0),
        m_current_pos(min_pos),
        m_current_trans(0),
        m_current_scale(0),
        m_current_ex_type(0)
    {
        srand ( time(NULL) );
    }
    map<string,float> operator()(){
        if(m_current_example == m_max_num_examples){
            throw max_example_reached_exception();
        }
        if(m_current_pos == m_max_pos){
            m_current_pos = m_min_pos;
        }

        // samples translation and input type only once, and iterates over all positions
        float trans;
        float scale;
        int ex_type;
        if(m_current_pos == m_min_pos){
            scale =  1 + (drand48() * 2 * m_max_scale - m_max_scale);
            trans = drand48() * 2 * m_max_trans - m_max_trans;
            ex_type = (rand() % (m_max_ex_type - m_min_ex_type )) + m_min_ex_type;
            m_current_trans = trans;
            m_current_scale = scale;
            m_current_ex_type = ex_type;
        }
        else{
            trans = m_current_trans;
            ex_type = m_current_ex_type;
            scale = m_current_scale;
        }
        m_current_example++;
        int pos = m_current_pos;
        m_current_pos++;

        tensor_type example(cuv::extents[3][1][m_input_size]);
        example = 0.f;        
        morse_code morse(example, m_factor);
        morse.write_char(ex_type, 0, 0, pos);
        morse.write_char(ex_type, 1, 0, pos);
        morse.write_char(ex_type, 2, 0, pos);
        morse.translate_coordinates(1, 0, trans);
        morse.translate_coordinates(1, 0, scale);

        morse.translate_coordinates(2, 0, trans);
        morse.translate_coordinates(2, 0, trans);
        morse.translate_coordinates(2, 0, scale);
        morse.translate_coordinates(2, 0, scale);

        morse.write_from_coordinates();
        example = morse.get_data();

        m_input_x->data() = example[cuv::indices[0][cuv::index_range()][cuv::index_range()]];
        m_input_y->data() = example[cuv::indices[1][cuv::index_range()][cuv::index_range()]];
        map<string,float> param_map;
        param_map["translation"] = trans;
        param_map["scaling"] = scale;
        param_map["position"] = pos;
        param_map["input_type"] = ex_type;
        return param_map;
    }
};



int main(int argc, char **argv)
{
    std::string param_name;
    int device = 2;
    if(argc >=1)
        device = boost::lexical_cast<int>(argv[1]);
    if(argc >= 2)
       param_name = argv[2];
    // initialize cuv library
    cuv::initCUDA(device);
    cuv::initialize_mersenne_twister_seeds();
    unsigned int input_size=100,bs=  1000 , subsampling = 2, max_trans = 1, gauss_dist = 6, min_width = 10, max_width = 30, flag = 2, morse_factor = 6;

    float max_growing = 0.f;
    unsigned max_num_epochs = 500;

    unsigned int num_hidden = 5;

    unsigned int num_factors = 600;
    float sigma = gauss_dist / 3; 
    //float learning_rate = 0.001f;
    // generate random translation datas
    unsigned int data_set_size = 5000;
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
    monitor mon(false);
    mon.add(monitor::WP_SCALAR_EPOCH_STATS, ae.loss(),        "total loss");
    mon.add(monitor::WP_SINK,               ae.get_decoded_y(), "decoded");
    mon.add(monitor::WP_SINK,               ae.get_factor_x(), "factorx");
    mon.add(monitor::WP_SINK,               ae.get_factor_y(), "factory");
    mon.add(monitor::WP_SINK,               ae.get_input_x(), "x");
    mon.add(monitor::WP_SINK,               ae.get_input_y(), "y");
    mon.add(monitor::WP_SINK,               ae.get_encoded(), "encoded");
    mon.add(monitor::WP_SINK,               ae.get_teacher(), "teacher");

    // does minibatch learning on the train set
    train_phase( ds,  mon,  ae,  input_x, input_y,  teacher,  bs, input_size,  num_hidden,  params, max_num_epochs);
    
    // evaluates the test set
    test_phase( ds,  mon,  ae,  input_x, input_y,  teacher,  bs, input_size,  num_hidden, param_name);

    // makes multiple predictions and visualizes inputs, predictions, hidden units, and factors 
    //prediction_phase( ds,  mon,  ae,  input_x, input_y,  teacher,  bs, input_size,  num_hidden, num_factors);
    
    // does regression on hidden layer activations, classifies transformation
    regression(ds, bs);

    // log to the file the generated examples for invariance analysis

    //input_x->data().resize(cuv::extents[1][input_size]); 
    //input_y->data().resize(cuv::extents[1][input_size]); 
    
    //dumper a;
    
    //int num_examples = 100000;
    //morse_pat_gen m(input_x, input_y, -max_trans, max_trans,  0,  input_size, 0, 38, morse_factor, input_size, num_examples, max_growing);
    //a.generate_log_patterns(ae.get_encoded(), m, "invariance_test-" + param_name + ".txt"); 

    

    return 0;
}
