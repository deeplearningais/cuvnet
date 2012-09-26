#ifndef __GENERIC_AUTO_ENCODER_HPP__
#     define __GENERIC_AUTO_ENCODER_HPP__

#include <boost/tuple/tuple.hpp>
#include <cuv.hpp>

#include <cuvnet/ops.hpp>

namespace cuvnet{


/**
 * Base class for various versions of the auto-encoder.
 *
 * @ingroup models
 */
class generic_auto_encoder{
    public:
        /** 
         * this is the type of a `function', e.g. the output or the loss of
         * the autoencoder
         */
        typedef boost::shared_ptr<Op>     op_ptr;
        enum mode{ AEM_SUPERVISED, AEM_UNSUPERVISED };
    private:
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version) { 
                ar & m_input & m_encoded & m_decoded & m_reg_loss & m_rec_loss & m_loss & m_binary & m_mode;
            }
    protected:
        mode m_mode;        ///< whether we are in supervised/unsupervised mode now
        op_ptr m_input;     ///< the input of the auto-encoder (might be different from what is supplied in \c init())
        op_ptr m_encoded;   ///< the encoder function
        op_ptr m_decoded;   ///< the decoder function

        op_ptr m_reg_loss;  ///< regularization loss
        op_ptr m_rec_loss;  ///< reconstruction loss
        op_ptr m_loss;      ///< the total loss of the autoencoder

        bool m_binary; ///< if \c m_binary is true, use logistic loss

    public:

        /**
         * Accessor for input object
         */
        inline op_ptr input(){ return m_input; }

        /**
         * deinit clears the losses (and thus the function graph),
         * but should not touch any parameters.
         */
        virtual void deinit(){
            //std::cout << typeid(*this).name() << " -- m_loss.use_count():" << m_loss.use_count() << std::endl;
            if(m_loss){
                m_loss->detach_from_results();
                m_loss->detach_from_params();
            }
            m_loss.reset();

            if(m_reg_loss){
                m_reg_loss->detach_from_results();
                m_reg_loss->detach_from_params();
            }
            m_reg_loss.reset();

            if(m_rec_loss){
                m_rec_loss->detach_from_results();
                m_rec_loss->detach_from_params();
            }
            m_rec_loss.reset();
        }

        /**
         * switch to supervised mode.
         *
         * This can be used if e.g. different regularizers are
         * to be used during finetuning.
         *
         * loss and regularization loss will be resetted, so
         * that they have to be created again.
         *
         * @param m the new mode
         */
        virtual void switch_mode(mode m){
            if(m!=m_mode){
                if(m_loss){
                    m_loss->detach_from_results();
                    m_loss->detach_from_params();
                }
                m_loss.reset();

                if(m_reg_loss){
                    m_reg_loss->detach_from_results();
                    m_reg_loss->detach_from_params();
                }
                m_reg_loss.reset();
            }
        }


        /**
         * Determine the parameters to be learned during unsupervised training.
         *
         * All Ops in the returned vector should be instances of class \c Input
         * 
         * @return the parameters used in unsupervised training (i.e. pre-training)
         */
        virtual std::vector<Op*> unsupervised_params()=0;

        /**
         * Determine the parameters to be learned during supervised training
         *
         * All Ops in the returned vector should be instances of class \c Input
         *
         * @return the parameters used in supervised training (i.e. fine-tuning)
         */
        virtual std::vector<Op*>   supervised_params()=0;

        /**
         * Encoder
         *
         * @param inp input to be encoded
         * @return a function that encodes the input of the autoencoder
         */
        virtual op_ptr encode(op_ptr& inp)=0;

        /**
         * Decoder
         *
         * @param enc a function that is the encoder of the autoencoder
         * @return a function that decodes the encoded values
         */
        virtual op_ptr decode(op_ptr& enc)=0;

        /**
         * the total loss, including regularization.
         *
         * The loss depends on whether this is a binary auto-encoder or not
         * i.e. binary AE will use the log-loss, otherwise mean square loss
         * will be used.
         * 
         * @return a function that calculates the loss of the auto-encoder
         */
        virtual op_ptr& loss(){
            if(!m_loss){
                float lambda;
                boost::tie(lambda, m_reg_loss)  = regularize();
                if(!lambda || !m_reg_loss)
                    m_loss = m_rec_loss;
                else
                    m_loss = axpby(m_rec_loss, lambda, label("regularizer", m_reg_loss));
            }
            return m_loss;
        };

        /**
         * Regularizer (eg for monitoring)
         * @return a function that calculates the regularizer of the auto-encoder
         */
        virtual op_ptr& reg_loss(){ return m_reg_loss; };

        /**
         * Reconstruction loss (eg for monitoring)
         * @return a function that calculates the reconstruction loss of the auto-encoder
         */
        virtual op_ptr& rec_loss(){ return m_rec_loss; };

        /**
         * Reset the weights so that the model can be retrained (eg for Crossvalidation).
         */
        virtual void reset_weights(){};

        /**
         * Constructor
         *
         * @param input a function that generates the input of the auto encoder
         * @param binary if true, assume that input variables are bernoulli-distributed
         */
        generic_auto_encoder(bool binary)
            :m_mode(AEM_UNSUPERVISED)
            ,m_binary(binary)
        {
        }

        /**
         * @return whether auto-encoder works on binary (=bernoulli-distributed) data
         */
        bool binary()const{return m_binary;}

        /**
         * @return the encoded representation
         */
        op_ptr& get_encoded(){return  m_encoded;}

        /**
         * @return the decoded representation
         */
        op_ptr& get_decoded(){return  m_decoded;}
    

        /**
         * Determine the reconstruction loss.
         *
         * @param input a function that determines the input of the auto-encoder
         * @param decode a function that determines the (decoded) output of the auto-encoder
         * @return a function that measures the quality of reconstruction between \c input and \c decode
         */
        virtual 
        op_ptr reconstruction_loss(op_ptr& input, op_ptr& decode){
            if(!m_binary)  // squared loss
                return mean( sum_to_vec(pow(axpby(input, -1.f, decode), 2.f), 0) );
            else         // cross-entropy
            {
                return mean( sum_to_vec(neg_log_cross_entropy_of_logistic(input,decode),0));
            }
        }

        /**
         * Returns the (additive) regularizer for the auto-encoder.
         * Defaults to no regularization.
         */
        virtual boost::tuple<float,op_ptr> regularize(){
            return boost::make_tuple(0.f,op_ptr());
        }

        /**
         * construct encoder, decoder and loss; initialize weights.
         *
         */
        virtual void init(op_ptr input){
            m_input     = input;
            m_encoded   = label("encoded",encode(m_input));
            m_decoded   = label("decoded",decode(m_encoded));
            m_rec_loss  = label("rec_loss",reconstruction_loss(m_input, m_decoded));

            reset_weights();
        }


        /**
         * Set the batch size to a new value
         *
         * You can overload this, default is to do nothing.
         *
         * @param bs the new batchsize
         */
        virtual void set_batchsize(unsigned int bs){};

        /**
         * (Virtual) dtor
         */
        virtual ~generic_auto_encoder(){}
};
}

#endif /* __GENERIC_AUTO_ENCODER_HPP__ */
