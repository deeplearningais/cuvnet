/*
 * spn.cpp
 *
 *  Created on: Oct 17, 2013
 *      Author: hartman3
 */

#include "spn.hpp"
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/ops.hpp>
#include "initialization.hpp"
#include <cuvnet/op_utils.hpp>

namespace cuvnet
{
    namespace models
    {

    void spn_layer::reset_params(){
        cuv::fill(m_W->data(), 0.f);
        //initialize_dense_glorot_bengio(m_W, true);
    }
    
    spn_layer::spn_layer(spn_layer::op_ptr X, unsigned int size, unsigned int sub_size, int nStride, int nPoolFlt, float eps, bool hard_gradient, bool pooling, std::string name){
        op_group grp("spnlayer");
        determine_shapes(*X);
        std::cout << "input shape: " << X->result()->shape[0] << " x " <<  X->result()->shape[1] << " x " <<  X->result()->shape[2]<<  " x " <<  X->result()->shape[3] << std::endl;

        using namespace cuv;
        std::string m_name("spn_layer_W_");
        m_name.append(name);    
        m_W = input(cuv::extents[size][sub_size], m_name);
        
        boost::shared_ptr<cuvnet::monitor> mon(new monitor(true));

        if (pooling)
        {
            if (hard_gradient)
            {                    
                m_output = weighted_sub_tensor_op( local_pool(X, nPoolFlt /*mask size*/, nPoolFlt /*movement of mask*/, alex_conv::PT_SUM /*Type of Pooling*/), 
                                                   m_W, mon, size, nStride, sub_size, cuv::alex_conv::TO_WMAX_LOGSPACE, eps, true);       
            }else{
                m_output = weighted_sub_tensor_op(local_pool( X, nPoolFlt /*mask size*/, nPoolFlt /*movement of mask*/, alex_conv::PT_SUM /*Type of Pooling*/), 
                                                  m_W, mon, size, nStride, sub_size, cuv::alex_conv::TO_LOGWADDEXP_LOGSPACE, eps, true);
            }  
            
            }else{
                if ( hard_gradient )
                {
                    m_output = weighted_sub_tensor_op(X, m_W, mon, size, nStride, sub_size,  cuv::alex_conv::TO_WMAX_LOGSPACE, eps, true );
                 } else {
                    m_output = weighted_sub_tensor_op(X, m_W, mon, size, nStride, sub_size, cuv::alex_conv::TO_LOGWADDEXP_LOGSPACE, eps, true );
                 }
      }
    }

        std::vector<Op*> spn_layer::get_params(){
            std::vector<Op*> params;
            if(m_W)    params.push_back(m_W.get());
            return params;
        }

////////////////////////////////////////////////////////////////////////////////////// conv layer ////////////////////////////////////////////////////////////////////////////

    void spn_conv_layer::reset_params(){
        initialize_dense_glorot_bengio(m_W, true);
    }

    // Todo: checken ob X->result()->shape[0] tatsächlich nFiltChannels ist.
    // check
    /** First param of Convolve, the images, must have shape
    *
    *  nChannels x nPixels x nImages
    */
    spn_conv_layer::spn_conv_layer(spn_layer::op_ptr X, unsigned int size, int nStride, int nConvFlt){
        op_group grp("convlayer");
        determine_shapes(*X);
        std::cout << "input shape: " << X->result()->shape[0] << " x " <<  X->result()->shape[1] << " x " <<  X->result()->shape[2] << " x " <<  X->result()->shape[3] << std::endl;

        m_W = input(cuv::extents[X->result()->shape[0] /*nFiltChannels*/][nConvFlt /*nFiltPix*/][size /*nFilt*/], "conv_W");

        // define output function
        m_output = log(convolve(X, exp(1.f, m_W), false /*padding*/, -1 /*padding size (not used)*/, nStride, 1  /*ngroups*/));
    }


        std::vector<Op*> spn_conv_layer::get_params(){
            std::vector<Op*> params;
            if(m_W)    params.push_back(m_W.get());
            return params;
        }

////////////////////////////////////////////////////////////////////////////////// output layer ////////////////////////////////////////////////////////////////////////

    // todo output layer back prop step: initialize with ones
    void spn_out_layer::reset_params(){
        cuv::fill(m_W->data(), 0.f);
        //initialize_dense_glorot_bengio(m_W, true);
    }

    /*
     * Output layer implements the top prod layer and the root sum layer.
     *
     * */
    spn_out_layer::spn_out_layer(std::vector<spn_layer::op_ptr> X, input_ptr Y, unsigned int n_classes, float eps)
    {
        op_group grp("out_layer");
        //dertermine shapes of all inputs
        for (unsigned int i = 0; i <  X.size(); i++)
        {
            determine_shapes(*X[i]);
        }
        std::cout << "input shape: " << X[0]->result()->shape[0] << " x " <<  X[0]->result()->shape[1] << " x " <<  X[0]->result()->shape[2] << " x " <<  X[0]->result()->shape[3] << std::endl;

        m_W   = input(cuv::extents[n_classes], "spn_output_W");

        //(input has shape:) nClasses[parts x nModules x nImg)
        boost::shared_ptr<cuvnet::monitor> mon(new monitor(true));
        
        //TODO SUM first dim of ALL THE X ( sum mat to vec obviously does not work = /
        m_output = spn_output_op(concatenate_n(X, 0, n_classes), m_W, Y, mon, n_classes, eps);        
    }

        std::vector<Op*> spn_out_layer::get_params(){
            std::vector<Op*> params;
            if(m_W)    params.push_back(m_W.get());
            return params;
        }


///////////////////////////////////////////////////////////////////////////// Classifier ////////////////////////////////////////////////////////////////////////////////

    spn_classifier::spn_classifier(
            spn_classifier::input_ptr X,
            spn_classifier::input_ptr Y,
            const unsigned int inputSize,
            const unsigned int nParts,
            const unsigned int nClasses,
            const float eta,
            const float eps,
            const std::vector<bool> hard_gradient,
            const unsigned int n_sub_size,
            const int nConvFlt,
            const int nStride,
            const int nStrideConv,
            const int nPoolFlt,
            const float weight_decay,
            const bool conv_layer){
        
         m_conv_layer = conv_layer;
         // calculate number of layers
         unsigned int nLayer = 0;
         unsigned int width;
         if (m_conv_layer){
            width = ((inputSize*inputSize) / (nStrideConv * nStrideConv));
         } else {
            width = inputSize*inputSize;
        }
        
         while ( width > 1 )
         {
             nLayer++;
             width /= (nPoolFlt * nPoolFlt);
         }
         
         std::cout << "number of layers: " << nLayer << std::endl;
         
         unsigned int n_layer = nLayer ;// ( number of SPN layers ( + output layer))
         
         //disjunct decomposition
         unsigned int size = nParts * std::pow((float) nStride, (int) nLayer+1);
        
          //std::cout << "generating conv layer: " << std::endl;
          op_ptr o;
          // convolution layer
          if ( m_conv_layer){
              o = X;
            c_layer.resize(1);
            c_layer[0] = spn_conv_layer(o, size, nStrideConv, nConvFlt);
            register_submodel(c_layer[0]);
            o = c_layer[0].m_output;
          } else {
            o = log(X);   
          }
          
          //get op pointer for every class
          std::vector<op_ptr> class_o;
          class_o.resize(nClasses);

          //std::cout << "generating spn layers: " << std::endl;
          
          // set op pointer to conv layer
          for (unsigned int i = 0; i < nClasses; i++)
          {
              class_o[i] = o;
          }

          //set size of layers
          m_layers.resize(nClasses);
          for (unsigned int i = 0; i < nClasses; i++)
          {
              m_layers[i].resize(n_layer);
          }

          //calculate size of new layer
          size /= nStride;
          //set up first layer after conv (no pooling here)
          //std::cout << "creating layer  0" << std::endl;
          for (unsigned int c = 0; c < nClasses; c++)
          {
              unsigned int l = n_layer-1;
              m_layers[c][l] = spn_layer(class_o[c] /*opt ptr*/, size /*size*/ , n_sub_size, nStride, nPoolFlt, eps, hard_gradient[l], false, std::to_string(c));
              register_submodel(m_layers[c][l]);
              class_o[c] = m_layers[c][l].m_output;
          }

          // set up all layers
          for (int l = n_layer-2; l >= 0; l--)
          {
              //calculate size of new layer
              size /= nStride;

              // for every class
              for (unsigned int c = 0; c < nClasses; c++)
              {
                  //std::cout << "creating layer[ " << c << "][" << l << "]"  << std::endl;
                  m_layers[c][l] = spn_layer(class_o[c] /*opt ptr*/, size /*size*/ , n_sub_size, nStride, nPoolFlt, eps, hard_gradient[l], true, std::to_string((l+1) * nClasses +c));
                  register_submodel(m_layers[c][l]);
                  class_o[c] = m_layers[c][l].m_output;
              }
           }

          //std::cout << "generating output layer: " << std::endl;
           
          //std::cout << "creating output layer" << std::endl;
          // output layer
          o_layer.resize(1);
          o_layer[0] = spn_out_layer(class_o, Y /*label*/, nClasses, eps);
          register_submodel(o_layer[0]);
          o = o_layer[0].m_output;
          
          //std::cout << "creating monitor" << std::endl;
          
          //get monitor on s;
          boost::shared_ptr<cuvnet::monitor> S(new monitor(true));
          S->add(monitor::WP_SINK, o_layer[0].m_output, "S");
          
          //TODO set S for convolution layer (after implementing it ) 
          o_layer[0].set_S(S);
           // set reference to output for every SPN layer.
          for (unsigned int l = 0; l < nLayer; l++)
          {
              for (unsigned int c = 0; c < nClasses; c++)
              {
                 //std::cout << "setting S in layer[ " << c << "][" << l << "]"  << std::endl;
                 m_layers[c][l].set_S(S);   
             }
          }         

          if (n_layer != hard_gradient.size()){ 
              std::cout << "error, n_layer size wrong \n n_layer = " << n_layer << ", hard_gradient.size() = " << hard_gradient.size() << std::endl;
        }
          
          unsigned int n_params = nLayer * nClasses +1;
          //std::cout << "nLayer: " << nLayer << ", nClasses: " << nClasses << std::endl;
          if (m_conv_layer) n_params++;
          
          //set flags for hadrd / soft gradient
          //std::cout << "setting up inference flags" << std::endl;
          boost::shared_ptr<std::vector<bool> > Inference_type(new std::vector<bool>(n_params)); 
          if ( m_conv_layer) Inference_type->at(0) = false; // convolution layer can not have hard inference
            for (unsigned int i = 0; i < nLayer; i++){ 
            if (m_conv_layer){
                for (unsigned int c = 0; c < nClasses; c++){
                    Inference_type->at(i*nClasses + c + 1) = hard_gradient[i];
                }
            } else{
                for (unsigned int c = 0; c < nClasses; c++){
                    Inference_type->at(i*nClasses + c) = hard_gradient[i];
                }                
                }
            }

          //std::cout << "set flag for output layer" << std::endl;
          // classes layer can not yet have hard inference ( not implemented )
          Inference_type->at(n_params-1) = false; 

          //std::cout << "reset params" << std::endl;          
          //init all params
          reset_params();
          
          //std::cout << "setting results to S" << std::endl; 
          this->results = S;
          
          //generate spn gradient descent
          //std::cout << "generating gd" << std::endl; 
          boost::shared_ptr<spn_gradient_descent> gdo(new spn_gradient_descent(o, Y, 0 /*result*/, results /*monitor*/, get_params(), Inference_type, eta, weight_decay));
          //std::cout << "dumping dot file" << std::endl; 
          gdo->get_swiper().dump("bla.dot", true);
          this->gd = gdo;
          //std::cout << "register gd" << std::endl; 
          S->register_gd(*gd);
    }
    
     std::vector<Op*> spn_classifier::get_params(){
            std::vector<Op*> params;
            //convolution layer
            if ( m_conv_layer){ 
                params.push_back(c_layer[0].get_params().at(0));  
            }
            for (unsigned int c = 0; c < m_layers.size(); c++)
                for (unsigned int l = 0; l < m_layers[0].size(); l++)
                {            
                    params.push_back(m_layers[c][l].get_params().at(0));
                }
            
            //output layer        
            params.push_back(o_layer[0].get_params().at(0));  
            return params;
        }
    
    
     void spn_classifier::reset_params(){
            //convolution layer
            if ( m_conv_layer){
                c_layer[0].reset_params();
            }
            for (unsigned int c = 0; c < m_layers.size(); c++)
                for (unsigned int l = 0; l < m_layers[0].size(); l++)
                {
                    m_layers[c][l].reset_params();
                }
            //output layer
            o_layer[0].reset_params();
        }        

    }
}
