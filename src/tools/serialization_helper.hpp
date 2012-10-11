#ifndef __SERIALIZATION_HELPER_HPP__
#     define __SERIALIZATION_HELPER_HPP__

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <cuv/basics/io.hpp>
#include <cuvnet/op_io.hpp>

namespace cuvnet
{
    template<class T>
        void serialize_to_file(std::string file, const boost::shared_ptr<T> obj, int idx=-1, int every=-1){
            namespace bar= boost::archive;
            if(every >= 0 && (idx % every != 0))
                return;
            if(idx>=0)
                file = (boost::format(file) % idx).str();
            std::cout << "serializing model to:" << file << std::endl;
            std::ofstream f(file.c_str());
            bar::binary_oarchive oa(f);
            register_objects(oa);
            oa << obj;
        }

    template<class T>
        boost::shared_ptr<T> deserialize_from_file(std::string file, int idx=-1){
            namespace bar= boost::archive;
            if(idx>=0)
                file = (boost::format(file) % idx).str();
            std::ifstream f(file.c_str());
            bar::binary_iarchive ia(f);
            register_objects(ia);
            boost::shared_ptr<T>  obj;
            ia >> obj;
            return obj;
            return obj;
        }
}
#endif /* __SERIALIZATION_HELPER_HPP__ */
