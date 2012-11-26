#ifndef __OP_SOFTMAX_HPP__
#     define __OP_SOFTMAX_HPP__

#include <cuvnet/op.hpp>
#include <cuv/libs/opt/opt.hpp>

namespace cuvnet
{
    /**
     * calculates negative cross-entropy of logistic (pointwise).
     * 
     * \f$- x \log z - (1-x) \log(1-z)\f$, where \f$z = 1/(1-\exp(-y))\f$
     *
     * @ingroup Ops
     */
    class NegCrossEntropyOfLogistic
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            public:
                NegCrossEntropyOfLogistic(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 target
                 * @param p1 estimator
                 */
                NegCrossEntropyOfLogistic(result_t& p0, result_t& p1)
                    :Op(2,1){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                    }
        };
    /**
     * epsilon insensitive loss function.
     * 
     * \f$ \max(0, |y-\hat y| - \varepsilon)^2\f$.
     *
     * @note The derivative is only implemented w.r.t. y_hat!
     *
     * @ingroup Ops
     */
    class EpsilonInsensitiveLoss
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
                float m_sensitivity;
            public:
                EpsilonInsensitiveLoss(){} ///< for serialization
                /**
                 * ctor.
                 * @param sensitivity if 0, this is equivalent to mean squares 
                 * @param p0 y
                 * @param p1 y_hat
                 */
                EpsilonInsensitiveLoss(float sensitivity, result_t& p0, result_t& p1)
                    :Op(2,1), m_sensitivity(sensitivity){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_sensitivity;
                    }
        };
    /**
     * Hinge Loss or Squared Hinge Loss.
     * 
     * \f$ \max(0, yt-1)\f$ or \f$ \max(0, yt-1)^2\f$.
     *
     * For hinge loss, the target is typically either 1 or -1 (not 0...).
     *
     * @note The derivative is only implemented w.r.t. y!
     *
     * @ingroup Ops
     */
    class HingeLoss
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
                float m_margin;
                bool  m_squared;
            public:
                HingeLoss(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 y
                 * @param p1 y_hat
                 * @param squared if true, use squared hinge loss
                 */
                HingeLoss(result_t& p0, result_t& p1, bool squared)
                    :Op(2,1), m_margin(1.f), m_squared(squared){
                         add_param(0,p0);
                         add_param(1,p1);
                     }

                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_margin;
                    }
        };
    /**
     * calculates the elementwise KullbackLeibler-Divergence of two Bernoulli
     * variables given by their means.
     * 
     * \f$ x\log (x/y) + (1-x)\log\frac{1-x}{1-y} \f$
     *
     * @ingroup Ops
     */
    class BernoulliKullbackLeibler
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            public:
                BernoulliKullbackLeibler(){} ///< for serialization
                /**
                 * ctor.
                 * @param p1 y_hat
                 * @param p0 y
                 */
                BernoulliKullbackLeibler(result_t& p0, result_t& p1)
                    :Op(2,1), m_scalar(-1.f){
                         add_param(0,p0);
                         add_param(1,p1);
                     }
                /**
                 * ctor.
                 * @param scalar the constant value which is desired as average of y_hat
                 * @param p0 y_hat
                 */
                BernoulliKullbackLeibler(float scalar, result_t& p0)
                    :Op(1,1),m_scalar(scalar){
                         add_param(0,p0);
                     }
                void fprop();
                void bprop();

                void _determine_shapes();
            private:
                void fprop_1p();
                void fprop_2p();
                void bprop_1p();
                void bprop_2p();
                float m_scalar;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_scalar;
                    }
        };
    /**
     * Softmax activation function.
     *
     * \f[
     * f(x_i) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}
     * \f]
     *
     * where the sum over j is either over the rows or the columns of a
     * matrix, depending on the second parameter.
     *
     * @ingroup Ops
     */
    class Softmax
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_axis;
                value_ptr m_result;
            public:
                Softmax(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 the input
                 * @param axis the axis to softmax over (allowed: first or last)
                 */
                Softmax(result_t& p0, unsigned int axis)
                    :Op(1,1)
                    ,m_axis(1-axis)
                {
                    add_param(0,p0);
                }
                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };

    /**
     * Multinomial logistic loss.
     *
     * For some dataset \f$D=(X,Y)\f$
     * \f[
     * l(\theta, D) = 
     *  -\sum_{i=1}^{|X|}\left[
     *      \sum_k y_{ik}\hat y_{ik} - \log\sum_k\exp(\hat y_{ik})
     *     \right]
     * \f]
     *
     * where \f$\hat y\f$ is a function of \f$\theta\f$.
     *
     * the second result of this op is the softmaxed input.
     *
     * @note that the derivative is only implemented for y_hat.
     *
     */
    class MultinomialLogisticLoss
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                unsigned int m_axis;
                value_type m_minus_logaddexp;
                bool         m_softmaxed;
            public:
                MultinomialLogisticLoss(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 y_hat (estimator)
                 * @param p1 y (target)
                 * @param axis axis to softmax over (allowed: first or last)
                 */
                MultinomialLogisticLoss(result_t& p0, result_t& p1, unsigned int axis)
                    :Op(2,2)
                    ,m_axis(axis)
                {
                    assert(m_axis==0 || m_axis==1);
                    add_param(0,p0);
                    add_param(1,p1);
                }
                void fprop();
                void bprop();
            void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_axis;
                    }
    };
}
#endif /* __OP_SOFTMAX_HPP__ */
