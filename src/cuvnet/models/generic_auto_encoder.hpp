#ifndef __GENERIC_AUTO_ENCODER_HPP__
#     define __GENERIC_AUTO_ENCODER_HPP__

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <cuv.hpp>

#include <cuvnet/ops.hpp>

using namespace cuvnet;
namespace acc=boost::accumulators;


/**
 * this is the base class for various versions of the auto-encoder
 */
class generic_auto_encoder {
    public:
        /// this type is used for accumulating the results during learning
        typedef
            acc::accumulator_set<double,
            acc::stats<acc::tag::mean, acc::tag::variance(acc::lazy) > > acc_t;

        /** 
         * this is the type of a `function', e.g. the output or the loss of
         * the autoencoder
         */
        typedef boost::shared_ptr<Op>     op_ptr;
        typedef boost::shared_ptr<Sink>   sink_ptr;
    protected:
        acc_t s_rec_loss;   ///< reconstruction loss
        acc_t s_reg_loss;   ///< regularization loss
        acc_t s_total_loss; ///< total loss

        op_ptr m_input;     ///< the inputs of the AE
        op_ptr m_encoded;   ///< the encoder function
        op_ptr m_decoded;   ///< the decoder function

        op_ptr m_reg_loss;  ///< regularization loss
        op_ptr m_rec_loss;  ///< reconstruction loss
        op_ptr m_loss;      ///< the total loss of the autoencoder

        sink_ptr m_reg_loss_sink; ///< keeps regularization loss value for accumulating using accumulate_loss()
        sink_ptr m_rec_loss_sink; ///< keeps reconstruction loss value for accumulating using accumulate_loss()
        sink_ptr m_loss_sink;     ///< keeps loss value for accumulating using accumulate_loss()

        bool m_binary; ///< if \c m_binary is true, use logistic loss

    public:
        /** 
         * remembers how many epochs this was trained for, e.g. for training on
         * the trainval set
         */
        acc_t s_iters;       

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
         * Regularizer.
         *
         * Defaults to nothing.
         */
        virtual op_ptr regularize(){ return op_ptr(); }

        /**
         * Loss
         *
         * The loss depends on whether this is a binary auto-encoder or not
         * i.e. binary AE will use the log-loss, otherwise mean square loss
         * will be used.
         * 
         * @return a function that calculates the loss of the auto-encoder
         */
        virtual op_ptr& loss(){ return m_loss; };

        /**
         * Reset the weights so that the model can be retrained (e.g. for Crossvalidation).
         */
        virtual void reset_weights(){};

        /**
         * Constructor
         *
         * @param input a function that generates the input of the auto encoder
         * @param binary if true, assume that input variables are bernoulli-distributed
         */
        generic_auto_encoder(op_ptr input, bool binary)
            :m_input(input)
            ,m_binary(binary)
        {
        }

        /**
         * @return whether auto-encoder works on binary (=bernoulli-distributed) data
         */
        bool binary()const{return m_binary;}

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
         * construct encoder, decoder and loss; initialize weights.
         *
         * You should call this function at the end of the constructor of your
         * derived class.
         */
        void init(float regularization_strength=0.0f){
            m_encoded   = encode(m_input);
            m_decoded   = decode(m_encoded);
            m_rec_loss  = reconstruction_loss(m_input, m_decoded);
            m_reg_loss  = regularize();

            if(!m_reg_loss || regularization_strength == 0.0f){
                m_loss = m_rec_loss;
            }else{
                m_reg_loss_sink = sink(m_reg_loss);
                m_loss          = axpby(m_loss, regularization_strength, m_reg_loss);
            }

            m_rec_loss_sink = sink(m_rec_loss);
            m_loss_sink     = sink(m_loss);
            reset_weights();
        }

        /**
         * Accumulate the loss (e.g. after processing one batch) statistics
         */
        virtual void accumulate_loss(){ s_total_loss((float) m_loss_sink->cdata()[0]);}

        /**
         * Determine the number of iterations this model was trained for on average
         * This can be used to use the same number of iterations during
         * learning on the train+validation set as on training only when early
         * stopping was used during crossvalidation
         *
         * @return number of epochs
         */
        unsigned int    avg_iters()  { 
            if(acc::count(s_iters)==0) 
                return 0;
            return acc::mean(s_iters); 
        }

        /**
         * Reset the loss values (to be called after an epoch)
         */
        void reset_loss() {
            s_rec_loss   = acc_t();
            s_reg_loss   = acc_t();
            s_total_loss = acc_t();
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
         * log the process to stdout
         *
         * @param what a identifier
         * @param epoch current epoch number
         */
        virtual
        void log_loss(const char* what, unsigned int epoch) {
            std::cout << "\r"<<"epoch "<<epoch<<", perf "<<acc::mean(s_total_loss)<<", reg "<<acc::mean(s_reg_loss)<<", rec "<<acc::mean(s_rec_loss)<<std::flush;
        }

        /**
         * Current performance measure (average over epoch)
         *
         * @return average loss over epoch
         */
        float perf() {
            return acc::mean(s_total_loss);
        }
};

#endif /* __GENERIC_AUTO_ENCODER_HPP__ */
