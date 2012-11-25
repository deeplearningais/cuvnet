#ifndef __OP_IO_HPP__
#     define __OP_IO_HPP__

#include <string>

//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/serialization/string.hpp>
//#include <boost/serialization/shared_ptr.hpp>
//#include <boost/serialization/weak_ptr.hpp>
//#include <boost/serialization/vector.hpp>
//#include <boost/serialization/export.hpp>
#include <cuv/basics/io.hpp>
#include <cuvnet/ops.hpp>

#include "op.hpp"
namespace cuvnet
{

    /**
     * @ingroup serialization
     * serialize an Op to a string, which can later be used to recover it.
     *
     * @see str2op
     *
     * @param o the Op to be serialized
     * @return a string containing the serialized version of the Op
     */
    std::string op2str(Op::op_ptr& o);

    /**
     * @ingroup serialization
     * deserialize an Op from a string.
     *
     * @see op2str
     *
     * @param s a string containing the serialized version of an Op
     * @return the deserialized Op
     */
    Op::op_ptr str2op(const std::string& s);

    /**
     * @ingroup serialization
     * register all known Op types to a boost archive.
     *
     * This is necessary, since we use polymorphic pointers for Ops.
     * Note that you don't have to call this function when using 
     * \c op2str and \c str2op.
     *
     * @see op2str
     * @see str2op
     *
     * @param ar the boost archive.
     */
    template < typename Archive > 
        void register_objects(Archive & ar) 
        { 
            // infrastructure
            ar.template register_type<Input>();
            ar.template register_type<ParameterInput>();
            ar.template register_type<Pipe>();
            ar.template register_type<Sink>();
            ar.template register_type<DeltaSink>();

            // +, -, *, /
            ar.template register_type<Axpby>();
            ar.template register_type<AddScalar>();
            ar.template register_type<MultScalar>();
            ar.template register_type<SubtractFromScalar>();
            ar.template register_type<Multiply>();
            ar.template register_type<Prod>();

            ar.template register_type<Log>();
            ar.template register_type<Exp>();
            ar.template register_type<Pow>();

            // reductions
            ar.template register_type<Sum>();
            ar.template register_type<Mean>();

            // matrix-vector
            ar.template register_type<MatPlusVec>();
            ar.template register_type<SumMatToVec>();
            ar.template register_type<MatTimesVec>();

            // trigonometry
            ar.template register_type<Tanh>();
            ar.template register_type<Sin>();
            ar.template register_type<Cos>();
            ar.template register_type<Logistic>();
            ar.template register_type<Atan2>();

            // Loss functions
            ar.template register_type<NegCrossEntropyOfLogistic>();
            ar.template register_type<Softmax>();
            ar.template register_type<MultinomialLogisticLoss>();
            ar.template register_type<ClassificationLoss>();
            ar.template register_type<F2Measure>();
            ar.template register_type<EpsilonInsensitiveLoss>();
            ar.template register_type<HingeLoss>();

            // convolutions
            ar.template register_type<Convolve>();
            ar.template register_type<ReorderForConv>();
            ar.template register_type<ReorderFromConv>();
            ar.template register_type<LocalPooling>();
            ar.template register_type<ResponseNormalization>();
            ar.template register_type<ResponseNormalizationCrossMaps>();
            ar.template register_type<ContrastNormalization>();
            ar.template register_type<SeparableFilter>();
            ar.template register_type<ResizeBilinear>();

            // misc
            ar.template register_type<Identity>();
            ar.template register_type<Noiser>();
            ar.template register_type<Flatten>();
            ar.template register_type<Reshape>();
            ar.template register_type<Abs>();
            ar.template register_type<RowSelector>();
            ar.template register_type<RectifiedLinear>();
            ar.template register_type<Printer>();
        } 
}

// force linking with the cpp file
namespace dummy { struct bogus { static int bogus_method(); }; } 
static int bogus_variable = dummy::bogus::bogus_method();


#endif /* __OP_IO_HPP__ */
