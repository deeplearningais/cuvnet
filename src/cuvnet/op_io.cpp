#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

#include "op_io.hpp"
#include <cuvnet/ops.hpp>


using namespace cuvnet;
using std::printf;


BOOST_CLASS_EXPORT(Op);
BOOST_CLASS_EXPORT(::cuvnet::detail::op_param<Op::value_type> );
BOOST_CLASS_EXPORT(::cuvnet::detail::op_result<Op::value_type> );
BOOST_CLASS_EXPORT(Input);
BOOST_CLASS_EXPORT(ParameterInput);
BOOST_CLASS_EXPORT(Sink);
BOOST_CLASS_EXPORT(Identity);
BOOST_CLASS_EXPORT(Axpby);
BOOST_CLASS_EXPORT(Multiply);
BOOST_CLASS_EXPORT(SubtractFromScalar);
BOOST_CLASS_EXPORT(Log);
BOOST_CLASS_EXPORT(Pow);
BOOST_CLASS_EXPORT(Prod);
BOOST_CLASS_EXPORT(Tanh);
BOOST_CLASS_EXPORT(Logistic);
BOOST_CLASS_EXPORT(NegCrossEntropyOfLogistic);
BOOST_CLASS_EXPORT(Noiser);
BOOST_CLASS_EXPORT(Sum);
BOOST_CLASS_EXPORT(SumMatToVec);
BOOST_CLASS_EXPORT(Mean);
BOOST_CLASS_EXPORT(MatPlusVec);
BOOST_CLASS_EXPORT(Convolve);
BOOST_CLASS_EXPORT(ReorderForConv);
BOOST_CLASS_EXPORT(Flatten);
BOOST_CLASS_EXPORT(Reshape);
BOOST_CLASS_EXPORT(Softmax);
BOOST_CLASS_EXPORT(MultinomialLogisticLoss);



namespace cuvnet{
    std::string op2str(Op::op_ptr& o){
        namespace bar=boost::archive;
        std::ostringstream ss;
        {
            bar::binary_oarchive oa(ss);
            oa << o;
        }
        return ss.str();
    }
    Op::op_ptr str2op(const std::string&s){
        namespace bar=boost::archive;
        std::istringstream ss(s);
        Op::op_ptr o;
        bar::binary_iarchive ia(ss);
        ia >> o;
        return o;
    }
    
}

int dummy::bogus::bogus_method() { return 0;}
