#ifndef __OP_IO_HPP__
#     define __OP_IO_HPP__

#include <string>

//#include <boost/archive/binary_iarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/serialization/string.hpp>
//#include <boost/serialization/shared_ptr.hpp>
//#include <boost/serialization/weak_ptr.hpp>
//#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
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

        } 
}

// force linking with the cpp file
//namespace dummy { struct bogus { static int bogus_method(); }; } 
//static int bogus_variable = dummy::bogus::bogus_method();

// infrastructure
BOOST_CLASS_EXPORT_KEY(cuvnet::Input);
BOOST_CLASS_EXPORT_KEY(cuvnet::ParameterInput);
BOOST_CLASS_EXPORT_KEY(cuvnet::Pipe);
BOOST_CLASS_EXPORT_KEY(cuvnet::Sink);
BOOST_CLASS_EXPORT_KEY(cuvnet::DeltaSink);

// +, -, *, /
BOOST_CLASS_EXPORT_KEY(cuvnet::Axpby);
BOOST_CLASS_EXPORT_KEY(cuvnet::AddScalar);
BOOST_CLASS_EXPORT_KEY(cuvnet::ScalarLike);
BOOST_CLASS_EXPORT_KEY(cuvnet::MultScalar);
BOOST_CLASS_EXPORT_KEY(cuvnet::SubtractFromScalar);
BOOST_CLASS_EXPORT_KEY(cuvnet::Multiply);
BOOST_CLASS_EXPORT_KEY(cuvnet::Prod);

BOOST_CLASS_EXPORT_KEY(cuvnet::Log);
BOOST_CLASS_EXPORT_KEY(cuvnet::Exp);
BOOST_CLASS_EXPORT_KEY(cuvnet::Pow);

// reductions
BOOST_CLASS_EXPORT_KEY(cuvnet::Sum);
BOOST_CLASS_EXPORT_KEY(cuvnet::Mean);

// matrix-vector
BOOST_CLASS_EXPORT_KEY(cuvnet::MatPlusVec);
BOOST_CLASS_EXPORT_KEY(cuvnet::SumMatToVec);
BOOST_CLASS_EXPORT_KEY(cuvnet::MatTimesVec);

// trigonometry
BOOST_CLASS_EXPORT_KEY(cuvnet::Tanh);
BOOST_CLASS_EXPORT_KEY(cuvnet::Sin);
BOOST_CLASS_EXPORT_KEY(cuvnet::Cos);
BOOST_CLASS_EXPORT_KEY(cuvnet::Logistic);
BOOST_CLASS_EXPORT_KEY(cuvnet::Atan2);

// Loss functions
BOOST_CLASS_EXPORT_KEY(cuvnet::NegCrossEntropyOfLogistic);
BOOST_CLASS_EXPORT_KEY(cuvnet::Softmax);
BOOST_CLASS_EXPORT_KEY(cuvnet::MultinomialLogisticLoss);
BOOST_CLASS_EXPORT_KEY(cuvnet::MultinomialLogisticLoss2);
BOOST_CLASS_EXPORT_KEY(cuvnet::ClassificationLoss);
BOOST_CLASS_EXPORT_KEY(cuvnet::F2Measure);
BOOST_CLASS_EXPORT_KEY(cuvnet::EpsilonInsensitiveLoss);
BOOST_CLASS_EXPORT_KEY(cuvnet::HingeLoss);

// convolutions
BOOST_CLASS_EXPORT_KEY(cuvnet::Convolve);
BOOST_CLASS_EXPORT_KEY(cuvnet::ReorderForConv);
BOOST_CLASS_EXPORT_KEY(cuvnet::ReorderFromConv);
BOOST_CLASS_EXPORT_KEY(cuvnet::LocalPooling);
BOOST_CLASS_EXPORT_KEY(cuvnet::ResponseNormalization);
BOOST_CLASS_EXPORT_KEY(cuvnet::ResponseNormalizationCrossMaps);
BOOST_CLASS_EXPORT_KEY(cuvnet::ContrastNormalization);
BOOST_CLASS_EXPORT_KEY(cuvnet::SeparableFilter);
BOOST_CLASS_EXPORT_KEY(cuvnet::SeparableFilter1d);
BOOST_CLASS_EXPORT_KEY(cuvnet::ResizeBilinear);
BOOST_CLASS_EXPORT_KEY(cuvnet::Tuplewise_op);
BOOST_CLASS_EXPORT_KEY(cuvnet::Subtensor);
BOOST_CLASS_EXPORT_KEY(cuvnet::BedOfNails);

// misc
BOOST_CLASS_EXPORT_KEY(cuvnet::Identity);
BOOST_CLASS_EXPORT_KEY(cuvnet::Noiser);
BOOST_CLASS_EXPORT_KEY(cuvnet::Flatten);
BOOST_CLASS_EXPORT_KEY(cuvnet::Reshape);
BOOST_CLASS_EXPORT_KEY(cuvnet::Concatenate);
BOOST_CLASS_EXPORT_KEY(cuvnet::Abs);
BOOST_CLASS_EXPORT_KEY(cuvnet::RowSelector);
BOOST_CLASS_EXPORT_KEY(cuvnet::RectifiedLinear);
BOOST_CLASS_EXPORT_KEY(cuvnet::Printer);

#ifndef NO_THEANO_WRAPPERS
//theano ops
BOOST_CLASS_EXPORT_KEY(cuvnet::FlipDims);
BOOST_CLASS_EXPORT_KEY(cuvnet::ShuffleDim);
BOOST_CLASS_EXPORT_KEY(cuvnet::Convolve2dTheano);
#endif

BOOST_CLASS_EXPORT_KEY(cuvnet::Sum_Out_Dim);
BOOST_CLASS_EXPORT_KEY(cuvnet::Upscale);
#endif /* __OP_IO_HPP__ */
