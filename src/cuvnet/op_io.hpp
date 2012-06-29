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
    std::string op2str(Op::op_ptr&);
    Op::op_ptr str2op(const std::string&);

    template < typename Archive > 
        void register_objects(Archive & ar) 
        { 
            // infrastructure
            ar.template register_type<Input>();
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
            ar.template register_type<EpsilonInsensitiveLoss>();

            // convolutions
            ar.template register_type<Convolve>();
            ar.template register_type<ReorderForConv>();
            ar.template register_type<ReorderFromConv>();
            ar.template register_type<LocalPooling>();

            // misc
            ar.template register_type<Identity>();
            ar.template register_type<Noiser>();
            ar.template register_type<Flatten>();
            ar.template register_type<Reshape>();
            ar.template register_type<Abs>();
            ar.template register_type<RowSelector>();
        } 
}

// force linking with the cpp file
namespace dummy { struct bogus { static int bogus_method(); }; } 
static int bogus_variable = dummy::bogus::bogus_method();


#endif /* __OP_IO_HPP__ */
