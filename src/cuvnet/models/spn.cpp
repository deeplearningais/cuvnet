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
#include "/home/stud/hartman3/local/checkout/thesis_hartman3/src/npc.hpp"



namespace cuvnet
{
    namespace models
    {

    void spn_layer::reset_params(){
        cuv::fill(m_W->data(), -5.f);
        //initialize_dense_glorot_bengio(m_W, true);
    }
    
    spn_layer::spn_layer(spn_layer::op_ptr X, unsigned int size, unsigned int sub_size, int nStride, int nPoolFlt, float eps, bool hard_gradient, bool pooling, std::string name){
        op_group grp("spnlayer");
        determine_shapes(*X);
       // std::cout << "input shape: " << X->result()->shape[0] << " x " <<  X->result()->shape[1] << " x " <<  X->result()->shape[2]<<  " x " <<  X->result()->shape[3] << std::endl;

        using namespace cuv;
        std::string m_name("spn_layer_W_");
        m_name.append(name);    
        m_W = input(cuv::extents[size][sub_size], m_name);
        cuv::fill(m_W->data(), -2.f);

        
        boost::shared_ptr<cuvnet::monitor> mon(new monitor(true));

        if (pooling)
        {
            if (hard_gradient)
            {                    
                m_output = weighted_sub_tensor_op( local_pool(X, nPoolFlt , nPoolFlt , alex_conv::PT_SUM ), 
                                                   m_W, mon, size, nStride, sub_size, cuv::alex_conv::TO_WMAX_LOGSPACE, eps, true);       
            }else{
                m_output = weighted_sub_tensor_op(local_pool( X, nPoolFlt , nPoolFlt, alex_conv::PT_SUM ), 
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
        //init or load weights
        if ( !m_load_weights)
            initialize_dense_glorot_bengio(m_W, true);
      }

    spn_conv_layer::spn_conv_layer(spn_layer::op_ptr X, unsigned int size, int nStride, int nConvFlt, bool load_weights){
        op_group grp("convlayer");
        determine_shapes(*X);
       // std::cout << "input shape: " << X->result()->shape[0] << " x " <<  X->result()->shape[1] << " x " <<  X->result()->shape[2] << " x " <<  X->result()->shape[3] << std::endl;
        m_W = input(cuv::extents[X->result()->shape[0] ][nConvFlt ][size ], "conv_W");
        //use pre learned filter?
        if(load_weights){
            m_load_weights = true;
            std::cout << "loading weights from file" << std::endl;
            m_W->data() = load_npc<float, cuv::dev_memory_space>("/home/stud/hartman3/local/checkout/thesis_hartman3/src/filter_banks/psi_filters-7-morlet-00-s2_x_t4_asqrt2.bin");
        } else {
            cuv::fill_rnd_uniform(m_W->data());
        }
        
        
        // define output function
        m_output = log(convolve(X, log_add_exp(m_W), false, -1 , nStride, 1 ));
    }

//TODO if rein, für load params
        std::vector<Op*> spn_conv_layer::get_params(){
            std::vector<Op*> params;
            params.push_back(m_W.get());
            return params;
        }

////////////////////////////////////////////////////////////////////////////////// output layer ////////////////////////////////////////////////////////////////////////

    // todo output layer back prop step: initialize with ones
    void spn_out_layer::reset_params(){
        cuv::fill(m_W->data(), -5.f);
        //initialize_dense_glorot_bengio(m_W, true);
    }


    spn_out_layer::spn_out_layer(std::vector<spn_layer::op_ptr> X, input_ptr Y, unsigned int n_classes, float eps)
    {
        op_group grp("out_layer");
        //dertermine shapes of all inputs
        for (unsigned int i = 0; i <  X.size(); i++)
        {
            determine_shapes(*X[i]);
        }
        //std::cout << "input shape: " << X[0]->result()->shape[0] << " x " <<  X[0]->result()->shape[1] << " x " <<  X[0]->result()->shape[2] << " x " <<  X[0]->result()->shape[3] << std::endl;

        m_W   = input(cuv::extents[n_classes], "spn_output_W");
        cuv::fill(m_W->data(), 0.f);

        //(input has shape:) nClasses[parts x nModules x nImg)
        boost::shared_ptr<cuvnet::monitor> mon(new monitor(true));

        unsigned int ndim = X[0]->result()->shape.size();
        unsigned int batch_size = X[0]->result()->shape[ndim -1];        
        
        for (unsigned int i = 0; i < X.size(); i++){
            X[i] = sum(X[i], 0);
        }

        m_root =  spn_output_op(concatenate(X, 0), m_W, Y, mon, n_classes, eps);
        m_output = reshape(m_root, cuv::extents[batch_size]);      
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
            const bool load_filter,
            const bool rescale_weights){
        
        std::cout << "load filter " << load_filter << std::endl;
         // calculate number of layers
         unsigned int nLayer = 0;
         unsigned int width;
         width = ((inputSize*inputSize) / (nStrideConv * nStrideConv));

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
              o = X;
            c_layer.resize(1);
            c_layer[0] = spn_conv_layer(o, size, nStrideConv, nConvFlt, load_filter);
            register_submodel(c_layer[0]);
            o = c_layer[0].m_output;
          
          //get op pointer for every class
          std::vector<op_ptr> class_o;
          class_o.resize(nClasses);
          
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
          for (unsigned int c = 0; c < nClasses; c++)
          {
              unsigned int l = n_layer-1;
              m_layers[c][l] = spn_layer(class_o[c] , size  , n_sub_size, nStride, nPoolFlt, eps, hard_gradient[l], false, std::to_string(c));
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
                  m_layers[c][l] = spn_layer(class_o[c] , size , n_sub_size, nStride, nPoolFlt, eps, hard_gradient[l], true, std::to_string((l+1) * nClasses +c));
                  register_submodel(m_layers[c][l]);
                  class_o[c] = m_layers[c][l].m_output;
              }
           }

          // output layer
          o_layer.resize(1);
          o_layer[0] = spn_out_layer(class_o, Y, nClasses, eps);
          register_submodel(o_layer[0]);
          o = o_layer[0].m_output;
                   
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
                 m_layers[c][l].set_S(S);   
             }
          }         

          if (n_layer != hard_gradient.size()){ 
              throw std::runtime_error( "error, n_layer size wrong ");
        }
          
          unsigned int n_params = nLayer * nClasses +1;
          n_params++;
          
          //set flags for hadrd / soft gradient
          boost::shared_ptr<std::vector<bool> > Inference_type(new std::vector<bool>(n_params)); 
          Inference_type->at(0) = false; // convolution layer can not have hard inference
            for (unsigned int i = 0; i < nLayer; i++){ 
                for (unsigned int c = 0; c < nClasses; c++){
                    Inference_type->at(i*nClasses + c + 1) = hard_gradient[i];
                }
            }

          // classes layer can not yet have hard inference ( not implemented )
          Inference_type->at(n_params-1) = false; 
 
          //init all params
          reset_params();
          this->results = S;
          
          //generate spn gradient descent
          boost::shared_ptr<spn_gradient_descent> gdo(new spn_gradient_descent(o, X, Y, 0 , results , get_params(), Inference_type, eta, rescale_weights, weight_decay));
          
          //dump dot file 
          gdo->get_swiper().dump("bla.dot", true);
          
          this->gd = gdo;
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




