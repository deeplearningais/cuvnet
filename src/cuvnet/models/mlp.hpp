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
            public:
                input_ptr m_W, m_bias;
                op_ptr m_output, m_linear_output;
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
                        ar & boost::serialization::base_object<model>(*this);
                        ar & m_output & m_W & m_bias;
                    };
        };

        struct mlp_classifier
            : public metamodel<model>{
                private:
                    typedef boost::shared_ptr<ParameterInput> input_ptr;
                    typedef boost::shared_ptr<Op> op_ptr;
                    std::vector<mlp_layer> m_layers;
                    logistic_regression m_logreg;
                public:
                    mlp_classifier(input_ptr X, input_ptr Y, std::vector<unsigned int> hlsizes);
                    virtual ~mlp_classifier(){}
                private:
                    friend class boost::serialization::access;
                    template<class Archive>
                        void serialize(Archive& ar, const unsigned int version) {
                            ar & boost::serialization::base_object<metamodel<model> >(*this);;
                            ar & m_layers & m_logreg;
                        };
            };
    }
}
BOOST_CLASS_EXPORT_KEY(cuvnet::models::mlp_layer);
BOOST_CLASS_EXPORT_KEY(cuvnet::models::mlp_classifier);

#endif /* __CUVNET_MODELS_MLP_HPP__ */
