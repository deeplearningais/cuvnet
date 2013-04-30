#ifndef __CUVNET_MODELS_MLP_HPP__
#     define __CUVNET_MODELS_MLP_HPP__

#include <cuvnet/ops/input.hpp>
#include "models.hpp"
#include "logistic_regression.hpp"

namespace cuvnet
{
    namespace models
    {
        struct mlp_layer
        : public model{
            private:
                typedef boost::shared_ptr<ParameterInput> input_ptr;
                typedef boost::shared_ptr<Op> op_ptr;
                input_ptr m_W, m_bias;
            public:
                op_ptr m_output;
                /**
                 * ctor.
                 * @param X input to the hidden layer
                 * @param size size of the hidden layer.
                 */
                mlp_layer(op_ptr X, unsigned int size);
                mlp_layer(){} ///< default ctor for serialization
                virtual std::vector<Op*> get_params();
                virtual void reset_params();
                virtual ~mlp_layer(){}
            private:
                friend class boost::serialization::access;                                                                 
                template<class Archive>                                                                                    
                    void serialize(Archive& ar, const unsigned int version) { 
                        ar & boost::serialization::base_object<boost::enable_shared_from_this<model> >(*this);;
                        ar & m_output & m_W & m_bias;
                    };
        };

        struct mlp
            : public metamodel{
                private:
                    typedef boost::shared_ptr<ParameterInput> input_ptr;
                    typedef boost::shared_ptr<Op> op_ptr;
                    std::vector<mlp_layer> m_layers;
                    logistic_regression m_logreg;
                public:
                    mlp(input_ptr X, input_ptr Y, std::vector<unsigned int> hlsizes);
                    virtual ~mlp(){}
                private:
                    friend class boost::serialization::access;                                                                 
                    template<class Archive>                                                                                    
                        void serialize(Archive& ar, const unsigned int version) { 
                            ar & boost::serialization::base_object<boost::enable_shared_from_this<metamodel> >(*this);;
                        };
            };
    }
}

#endif /* __CUVNET_MODELS_MLP_HPP__ */
