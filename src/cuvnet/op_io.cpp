#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/export.hpp>

#include "op_io.hpp"
#include <cuvnet/ops.hpp>


using namespace cuvnet;
using std::printf;


namespace cuvnet{
    std::string op2str(Op::op_ptr& o){
        namespace bar=boost::archive;
        std::ostringstream ss;
        {
            bar::binary_oarchive oa(ss);
            register_objects(oa);
            oa << o;
        }
        return ss.str();
    }
    Op::op_ptr str2op(const std::string&s){
        namespace bar=boost::archive;
        std::istringstream ss(s);
        Op::op_ptr o;
        bar::binary_iarchive ia(ss);
        register_objects(ia);
        ia >> o;
        return o;
    }
    
}

// infrastructure
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Input);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ParameterInput);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Pipe);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Sink);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::DeltaSink);

// +, -, *, /
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Axpby);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::AddScalar);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ScalarLike);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::MultScalar);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::SubtractFromScalar);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Multiply);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Prod);

BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Log);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Exp);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Pow);

// reductions
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Sum);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Mean);

// matrix-vector
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::MatPlusVec);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::SumMatToVec);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::MatTimesVec);

// trigonometry
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Tanh);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Sin);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Cos);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Logistic);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Atan2);

// Loss functions
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::NegCrossEntropyOfLogistic);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Softmax);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::MultinomialLogisticLoss);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::MultinomialLogisticLoss2);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ClassificationLoss);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::F2Measure);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::EpsilonInsensitiveLoss);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::HingeLoss);

// convolutions
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Convolve);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ReorderForConv);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ReorderFromConv);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::LocalPooling);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ResponseNormalization);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ResponseNormalizationCrossMaps);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ContrastNormalization);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::SeparableFilter);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::SeparableFilter1d);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ResizeBilinear);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Tuplewise_op);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Subtensor);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::BedOfNails);

// misc
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Identity);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Noiser);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Flatten);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Reshape);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Concatenate);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Abs);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::RowSelector);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::RectifiedLinear);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Printer);

#ifndef NO_THEANO_WRAPPERS
//theano ops
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::FlipDims);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::ShuffleDim);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Convolve2dTheano);
#endif
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Upscale);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::Sum_Out_Dim);
//int dummy::bogus::bogus_method() { return 0;}
