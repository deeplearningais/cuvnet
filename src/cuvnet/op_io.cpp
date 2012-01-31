#include "op_io.hpp"
#include <cuvnet/ops.hpp>
#include <boost/serialization/export.hpp>


using namespace cuvnet;
using std::printf;


//BOOST_CLASS_EXPORT_KEY(Op);
//BOOST_CLASS_EXPORT_KEY(::cuvnet::detail::op_param<Op::value_type> );
//BOOST_CLASS_EXPORT_KEY(::cuvnet::detail::op_result<Op::value_type> );
//BOOST_CLASS_EXPORT_KEY(Input);
//BOOST_CLASS_EXPORT_KEY(Output);
//BOOST_CLASS_EXPORT_KEY(Identity);
//BOOST_CLASS_EXPORT_KEY(Axpby);
//BOOST_CLASS_EXPORT_KEY(Multiply);
//BOOST_CLASS_EXPORT_KEY(SubtractFromScalar);
//BOOST_CLASS_EXPORT_KEY(Log);
//BOOST_CLASS_EXPORT_KEY(Pow);
//BOOST_CLASS_EXPORT_KEY(Prod);
//BOOST_CLASS_EXPORT_KEY(Tanh);
//BOOST_CLASS_EXPORT_KEY(Logistic);
//BOOST_CLASS_EXPORT_KEY(NegCrossEntropyOfLogistic);
//BOOST_CLASS_EXPORT_KEY(Noiser);
//BOOST_CLASS_EXPORT_KEY(Sum);
//BOOST_CLASS_EXPORT_KEY(SumMatToVec);
//BOOST_CLASS_EXPORT_KEY(Mean);
//BOOST_CLASS_EXPORT_KEY(MatPlusVec);
//BOOST_CLASS_EXPORT_KEY(Convolve);
//BOOST_CLASS_EXPORT_KEY(ReorderForConv);
//BOOST_CLASS_EXPORT_KEY(Flatten);
//BOOST_CLASS_EXPORT_KEY(Reshape);
//BOOST_CLASS_EXPORT_KEY(Softmax);
//BOOST_CLASS_EXPORT_KEY(MultinomialLogisticLoss);



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
