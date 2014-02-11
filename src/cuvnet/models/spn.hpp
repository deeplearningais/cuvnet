/*
 * spn.h
 *
 *  Created on: Oct 17, 2013
 *      Author: hartman3
 */

#ifndef SPN_HPP_
#define SPN_HPP_



#include <cuvnet/ops/input.hpp>
#include <cuvnet/models/models.hpp>
#include <cuvnet/models/initialization.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops.hpp>
#include <math.h>
#include <cuvnet/tools/gradient_descent.hpp>

namespace cuvnet
{   
    namespace models
    {

    struct spn_layer
    : public model{
        public:
        typedef Op::result_t      result_t;
        typedef boost::shared_ptr<ParameterInput> input_ptr;
        typedef boost::shared_ptr<Weighted_Sub_Tensor_op> wst_op_ptr;
        typedef boost::shared_ptr<Op> op_ptr;

        private:
            input_ptr m_W;
            unsigned int nNeighbourhood, nStride, nPoolFlt;
        public:
            wst_op_ptr m_output;
            /**
             * ctor.
             * @param X input to the hidden layer
             * @param size of the sum layer.
             * @param pooling insert product layer before sum layer ( only false for first layer )
             * @param nSumNodeIn = 3: (odd) number of decomposition weighted by a sumNode ( overlapping, default it gets position and two neighbors )
             * @param nStride = 1: stride in polling layer
             * @param nPoolFlt = 3: size of pooling mask in product layer
             * @param eps small numeric constant to hold logAddExp and 1/S[x] stable
             * @param hard_gradient flag for hard gradient_descent
             * @param pooling flag for pooling opereation
             * @param name name of this node
             */
            spn_layer(op_ptr X, unsigned int size, unsigned int nNeighbourhood, int nStride, int nPoolFlt, float eps, bool hard_gradient = false, bool pooling = true, std::string name = std::string(""));
            spn_layer(){} ///< default ctor for serialization
            virtual std::vector<Op*> get_params();
            virtual void reset_params();
            virtual ~spn_layer(){}
            
            void set_S(boost::shared_ptr<monitor> S){
                m_output->set_S(S);
            }

        private:
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) {
                    ar & boost::serialization::base_object<metamodel<model> >(*this);;
                    ar & m_output & m_W;
                };
    };

    struct spn_conv_layer
     : public model{
    private:
         typedef boost::shared_ptr<ParameterInput> input_ptr;
         typedef boost::shared_ptr<Op> op_ptr;
         input_ptr m_W;
         unsigned int nStride;
         bool m_load_weights = false;
     public:
         op_ptr m_output;
         /**
          * ctor.
          * @param X input to the convolution layer
          * @param size number of filters.
          * @param nStride = 1: stride in pooling layer
          * @param nConvFlt = 9: size of convolution mask in input mask (must be of size x*y, with x == y)
          */
         spn_conv_layer(op_ptr X, unsigned int size, int nStride = 1, int nConvFlt = 9, bool load_weights = false);
         spn_conv_layer(){} ///< default ctor for serialization
         virtual std::vector<Op*> get_params();
         virtual void reset_params();
         virtual ~spn_conv_layer(){}
     private:
         friend class boost::serialization::access;
         template<class Archive>
             void serialize(Archive& ar, const unsigned int version) {
                 ar & boost::serialization::base_object<metamodel<model> >(*this);;
                 ar & m_output & m_W;
             };
     };


     struct spn_out_layer
     : public model{
         private:
             typedef boost::shared_ptr<ParameterInput> input_ptr;
             typedef boost::shared_ptr<Op> op_ptr;
             typedef boost::shared_ptr<Spn_Output_Op> spn_op_ptr;
             input_ptr m_W;
         public:
             spn_op_ptr m_root;
             op_ptr m_output;
             /**
              * ctor.
              * @param X input to the hidden layer
              * @param Y labels
              * @param n_classes total number of classes in SPN
              * @param eps small numerical constant 
              */
             spn_out_layer(std::vector<op_ptr> X, input_ptr Y, unsigned int n_classes, float eps);
             spn_out_layer(){} ///< default ctor for serialization
             virtual std::vector<Op*> get_params();
             virtual void reset_params();
             virtual ~spn_out_layer(){}
             
             void set_S(boost::shared_ptr<monitor> S){
                m_root->set_S(S);
              }
         private:

             friend class boost::serialization::access;
             template<class Archive>
                 void serialize(Archive& ar, const unsigned int version) {
                     ar & boost::serialization::base_object<metamodel<model> >(*this);;
                     ar & m_output & m_W;
                 }
     };


    struct spn_classifier
        : public metamodel<model>{
            private:
                typedef boost::shared_ptr<ParameterInput> input_ptr;
                typedef boost::shared_ptr<Op> op_ptr;
                std::vector<std::vector<spn_layer> > m_layers;
                std::vector<spn_conv_layer> c_layer;
                std::vector<spn_out_layer> o_layer;
                boost::shared_ptr<monitor> results;
                bool m_conv_layer;
            public:
                boost::shared_ptr<spn_gradient_descent> gd;
                /**
                 * @param inputSize: size of the image ( assuming images of size w**2 )
                 * @param nParts: Number of parts (sum nodes in top layer of each class )
                 * @param nDecomp: number of decompositions added in each layer ( top to bottom size of layer l {0,.., n}: nParts * nDecomp**l )
                 * @param nClasses: number of classes
                 * @param eta: learn rate
                 * @param eps: small additive constant used to hold divisions numerical stable
                 * @param nNeighbourhood = 3: size of neighbourhood a sum node takes
                 * @param nConvFlt = 3: size of convolution mask in input mask
                 * @param nStride = 2: stride ( movement of filter in convolution and pooling )
                 * @param nStrideConv = 1: stride for convolution layer
                 * @param nPoolFlt = 3: size of pooling mask in all product layers
                 * @param weight_decay value for weight decay
                 * @param conv_layer flag to set convolution layer
                 */
                spn_classifier(input_ptr X, input_ptr Y, const unsigned int inputSize,
                                                         const unsigned int nParts,
                                                         const unsigned int nClasses,
                                                         const float eta,
                                                         const float eps,
                                                         const std::vector<bool> hard_gradient,
                                                         const unsigned int nNeighbourhood = 3,
                                                         const int nConvFlt = 3,
                                                         const int nStride = 2,
                                                         const int nStrideConv = 1,
                                                         const int nPoolFlt = 3,
                                                         const float weight_decay=0.f,
                                                         const bool load_filter = false,
                                                         const bool rescale_weights = true);
                virtual std::vector<Op*> get_params();
                virtual void reset_params();
                virtual ~spn_classifier(){}
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version) {
                        ar & boost::serialization::base_object<metamodel<model> >(*this);;
                        ar & m_layers;
                        ar & c_layer;
                        ar & o_layer;
                        ar & gd;
                    };
        };
        
    }

}

#endif /* SPN_HPP_ */
