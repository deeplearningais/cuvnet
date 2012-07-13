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

int dummy::bogus::bogus_method() { return 0;}
