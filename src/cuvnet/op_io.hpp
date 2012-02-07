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
            //ar.template register_type<Input>();
            //ar.template register_type<Output>();
            //ar.template register_type<Identity>();
            //ar.template register_type<Axpby>();
            //ar.template register_type<Multiply>();
            //ar.template register_type<SubtractFromScalar>();
            //ar.template register_type<Log>();
            //ar.template register_type<Pow>();
            //ar.template register_type<Prod>();
            //ar.template register_type<Tanh>();
            //ar.template register_type<Logistic>();
            //ar.template register_type<NegCrossEntropyOfLogistic>();
            //ar.template register_type<Noiser>();
            //ar.template register_type<Sum>();
            //ar.template register_type<SumMatToVec>();
            //ar.template register_type<Mean>();
            //ar.template register_type<MatPlusVec>();
            //ar.template register_type<Convolve>();
            //ar.template register_type<ReorderForConv>();
            //ar.template register_type<Flatten>();
            //ar.template register_type<Reshape>();
            //ar.template register_type<Softmax>();
            //ar.template register_type<MultinomialLogisticLoss>();
        } 
}

// force linking with the cpp file
namespace dummy { struct bogus { static int bogus_method(); }; } 
static int bogus_variable = dummy::bogus::bogus_method();


#endif /* __OP_IO_HPP__ */
