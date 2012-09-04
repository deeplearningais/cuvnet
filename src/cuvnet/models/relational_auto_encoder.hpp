#ifndef __RELATIONAL_AUTOENCODER_HPP__
#     define __RELATIONAL_AUTOENCODER_HPP__

#include <cuv.hpp>

#include <boost/assign.hpp>
#include <cuvnet/ops.hpp>

namespace cuvnet{
/**
 * Class for the relational auto-encoder.
 *
 * @ingroup models
 */
class relational_auto_encoder{
    public:
        /** 
         * this is the type of a `function', e.g. the output or the loss of
         * the autoencoder
         */
        typedef boost::shared_ptr<Op>     op_ptr;
        typedef boost::shared_ptr<ParameterInput> input_ptr;   ///< the input function e.g weights, bias
    protected:
        op_ptr m_input_x;           ///< the input x of the relational auto-encoder      
        op_ptr m_input_y;           ///< the input y of the relational auto-encoder      
        op_ptr m_teacher;           ///< the teacher of the relational auto-encoder      
        op_ptr m_encoded;           ///< the encoder function 
        op_ptr m_decoded_x;         ///< the decoder function when we want to reconstruct x from y
        op_ptr m_decoded_y;         ///< the decoder function when we want to reconstruct y from x
        op_ptr m_prediction;         ///< the prediction function
        op_ptr m_factor_x;          ///< the projection of x
        op_ptr m_factor_y;          ///< the projection of y
        op_ptr m_factor_h;          ///< the projection of h
        op_ptr m_factor_y_fx;       ///< the projection of input y to the matrix fx
        unsigned int m_hidden_dim;  ///< the number of hidden units
        bool m_binary;
        unsigned int m_num_factors; ///< the dimension of factors
        // these are the parametrs of the model
        input_ptr  m_fx, m_fy, m_fh, m_bias_x, m_bias_y, m_bias_h;

        op_ptr m_loss;      ///< the loss of the relational auto-encoder


    public: 
        /**
         * constructor
         * 
         * @param input the function that generates the input of the autoencoder
         * @param hidden_dim the number of dimensions of the hidden layer
         * @param binary if true, assumes inputs are bernoulli distributed
         */
        relational_auto_encoder(unsigned int hidden_dim, unsigned int num_factors, bool binary):
            m_hidden_dim(hidden_dim),
            m_binary(binary),
            m_num_factors(num_factors)
        {
        }

        /**
         * construct encoder, decoder and loss; initialize weights.
         *
         */
        void init(op_ptr input_x, op_ptr input_y, op_ptr teacher){
            m_input_x   = input_x;
            m_input_y   = input_y;
            m_teacher = teacher;
            
            // determines the dimension of input x and y and teacher
            m_input_x->visit(determine_shapes_visitor()); 
            unsigned int input_x_dim = m_input_x->result()->shape[1];
            m_input_y->visit(determine_shapes_visitor()); 
            unsigned int input_y_dim = m_input_y->result()->shape[1];

            // initialize weights and biases
            m_fx.reset(new ParameterInput(cuv::extents[input_x_dim][m_num_factors],"w_fx"));
            m_fy.reset(new ParameterInput(cuv::extents[input_y_dim][m_num_factors],"w_fy"));
            m_fh.reset(new ParameterInput(cuv::extents[m_hidden_dim][m_num_factors],"w_fh"));
            m_bias_h.reset(new ParameterInput(cuv::extents[m_hidden_dim],            "bias_h"));
            m_bias_x.reset(new ParameterInput(cuv::extents[input_x_dim],             "bias_x"));
            m_bias_y.reset(new ParameterInput(cuv::extents[input_y_dim],             "bias_y"));

            std::cout << " w_fx has dim: [" << input_x_dim << "][" << m_num_factors << "]" << std::endl;
            std::cout << " w_fy has dim: [" << input_y_dim << "][" << m_num_factors << "]" << std::endl;
            std::cout << " w_fh has dim: [" << m_hidden_dim << "][" << m_num_factors << "]" << std::endl;
            
            // calculates the projections of x and y
            m_factor_x = prod(m_input_x, m_fx);
            m_factor_y = prod(m_input_y, m_fy);
            m_factor_y_fx = prod(m_input_y, m_fx);
            
            // calculates encoder and projection of encoder
            m_encoded  = logistic(mat_plus_vec(
                               prod(m_factor_x * m_factor_y, m_fh, 'n', 't')
                               , m_bias_h, 1));
            m_factor_h = prod(m_encoded, m_fh);


            // calculates the prediction
            m_decoded_y = decode(m_factor_y_fx, m_factor_h, m_fy, m_bias_y);

            // calculates the reconstruction loss
            m_loss  = reconstruction_loss(m_teacher,  m_decoded_y);
    
            reset_weights();
        }
    
        
        /// \f$  h W^T + b_y \f$
        op_ptr  decode(op_ptr& factor, op_ptr& factor_h, input_ptr w, input_ptr b){ 
            return mat_plus_vec(
                    prod(factor * factor_h, w, 'n', 't')
                    , b, 1);
        }
        
        
       op_ptr get_decoded_x(){return m_decoded_x;} 
       op_ptr get_decoded_y(){return m_decoded_y;} 
       op_ptr get_factor_x(){return m_factor_x;} 
       op_ptr get_factor_y(){return m_factor_y;}
       op_ptr get_encoded(){return m_encoded;}
        

        /**
         * Determine the reconstruction loss.
         *
         * @param input_x a function that determines the input x of the relational auto-encoder
         * @param input_y a function that determines the input y of the relational auto-encoder
         * @param decode_x a function that determines the reconstruction of x of the relational auto-encoder
         * @param decode_y a function that determines the reconstruction of y of the relational auto-encoder
         * @return a function that measures the quality of reconstruction 
         */
       op_ptr reconstruction_loss(op_ptr& teacher, op_ptr& decode_y){
           if(!m_binary)  // squared loss
               return   mean( sum_to_vec(pow(axpby(decode_y, -1.f, teacher), 2.f), 0) );
           else         // cross-entropy
               return mean( sum_to_vec(neg_log_cross_entropy_of_logistic(teacher, decode_y),0));
       }
    

        /**
         * Loss
         *
         * The loss depends on whether this is a binary relational auto-encoder or not
         * i.e. binary will use the log-loss, otherwise mean square loss
         * will be used.
         * 
         * @return a function that calculates the loss of the relational auto-encoder
         */
        op_ptr& loss(){ return m_loss; }
        
        /**
         * initialize the weights with random numbers
         */
        virtual void reset_weights(){
            unsigned int input_dim_x = m_fx->data().shape(0);
            unsigned int factor_dim = m_fh->data().shape(1);
            unsigned int hidden_dim = m_fh->data().shape(0);
            float diff_x = 4.f*std::sqrt(6.f/(input_dim_x + factor_dim));
            float diff_h = 4.f*std::sqrt(6.f/(hidden_dim + factor_dim));
            
            cuv::fill_rnd_uniform(m_fx->data());
            cuv::fill_rnd_uniform(m_fy->data());
            cuv::fill_rnd_uniform(m_fh->data());
            diff_x *= 0.01;
            diff_h *= 0.01;
            m_fx->data() *= 2*diff_x;
            m_fx->data() -=   diff_x;
            m_fy->data() *= 2*diff_x;
            m_fy->data() -=   diff_x;
            m_fh->data() *= 2*diff_h;
            m_fh->data() -=   diff_h;
            
            m_fh->set_weight_decay_factor(1.f);
            m_fx->set_weight_decay_factor(1.f);
            m_fy->set_weight_decay_factor(1.f);
           
            //unsigned int num_hidd =  m_fh->data().shape(0);
            //unsigned int num_fact = m_fh->data().shape(1);
            //assert(num_fact % num_hidd == 0);
            //unsigned int stride = num_fact / num_hidd; 
            //cuv::tensor<float,cuv::host_memory_space> conv_kernel(cuv::extents[2*stride]);
            //cuv::tensor<float,cuv::host_memory_space> temp(cuv::extents[num_hidd][num_fact]);
            //conv_kernel = 1.f;
            //m_fh->data() = 0.f;
            //temp = 0.f;
            //int sum = 0;

            //for(unsigned int i = 0; i < num_hidd; i++){
            //  temp(i, i * stride) = 1.f;
            //  m_fh->data()(i, i * stride) = 1.f;
            //  for(unsigned int j = 0; j < num_fact; j++){
            //      for(unsigned int k = 0; k < 2*stride; k++){
            //          int index = j - (k - stride);
            //          if(index >= 0 && index < (int)num_fact){
            //              sum += m_fh->data()(i, index) * conv_kernel(k);
            //          }

            //      }
            //      temp(i,j) = sum;
            //      sum = 0;
            //      //std::cout << "  " << temp(i,j) ;
            //  }
            //  //std::cout << std::endl;

            //}
            //m_fh->data() = temp.copy();
            
            m_bias_h->data()   = 0.f;
            m_bias_y->data()   = 0.f;
            m_bias_x->data()   = 0.f;

            m_bias_h->set_weight_decay_factor(0.f);
            m_bias_x->set_weight_decay_factor(0.f);
            m_bias_y->set_weight_decay_factor(0.f);
        }

    /**
     * Determine the unsupervised parameters learned during training
     */
    std::vector<Op*> unsupervised_params(){
        return boost::assign::list_of(m_fx.get())(m_fy.get())(m_fh.get())(m_bias_h.get())(m_bias_y.get());
    }
    
    /**
     * Determine the supervised parameters learned during training
     */
    std::vector<Op*> supervised_params(){
        return boost::assign::list_of(m_fx.get())(m_fy.get())(m_fh.get())(m_bias_h.get());
    }
    
    /**
     * @return weight matrix fx
     */
    boost::shared_ptr<ParameterInput> get_fx(){return m_fx;} 

    /**
     * @return weight matrix fx
     */
    boost::shared_ptr<ParameterInput> get_fy(){return m_fy;} 

    /**
     * @return weight matrix fh
     */
    boost::shared_ptr<ParameterInput> get_fh(){return m_fh;} 

};


}

#endif /* __RELATIONAL_AUTOENCODER_HPP__
 */
