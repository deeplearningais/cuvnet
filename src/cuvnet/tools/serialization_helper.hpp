#ifndef __SERIALIZATION_HELPER_HPP__
#     define __SERIALIZATION_HELPER_HPP__

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include <cuv/basics/io.hpp>
#include <cuvnet/op_io.hpp>

namespace cuvnet
{
    /**
     * @ingroup serialization
     * Serialize whole models to a file.
     *
     * The file can get a version number. The method is intended to be used as
     * an after_epoch callback in gradient_descent:
     * 
     * @code
     * boost::shared_ptr<model> model_ptr = ...;
     * gradient_descent gd(...);
     * gd.after_epoch.connect(
     *    boost::bind(serialize_to_file<model>, 
     *       "model-%04d.ser", model_ptr, _1, 20));
     * @endcode
     * @see deserialize_from_file
     *
     * @param file the filename. If idx is given, the file must be format string accepting an integer, e.g. "model-%04d.ser".
     * @param obj the model to be serialized
     * @param idx a running number
     * @param every if given, we only save if \c idx % \c every == 0.
     */
    template<class T>
        void serialize_to_file(std::string file, const boost::shared_ptr<T> obj, int idx=-1, int every=-1){
            namespace bar= boost::archive;
            if(every >= 0 && (idx % every != 0))
                return;
            if(idx>=0)
                file = (boost::format(file) % idx).str();
            std::cout << "serializing model to: " << file << std::endl;
            std::ofstream f(file.c_str());
            bar::binary_oarchive oa(f);
            register_objects(oa);
            oa << obj;
        }

    /**
     * @ingroup serialization
     * Deserialize a whole model from a file.
     *
     * The file may have a version number.
     * @code
     * // load the model with index given by argv[1]
     * boost::shared_ptr<model> model_ptr =
     *   deserialize_from_file<model>("model-%04d.ser", 
     *       boost::lexical_cast<int>(argv[1]));
     * @endcode
     *
     * @see serialize_to_file
     *
     * @param file the filename. If idx is given, the file must be format string accepting an integer, e.g. "model-%04d.ser".
     * @param idx optional, if given the filename is assumed to be a format string which accepts idx
     * @return deserialized model
     */
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
        }


    /**
     * Same as serialize_to_file, but with the possibility of registering own classes with the archive in a callback.
     * @see serialize_to_file
     *
     * @param file the filename. If idx is given, the file must be format string accepting an integer, e.g. "model-%04d.ser".
     * @param obj the model to be serialized
     * @param r a callback taking a boost archive
     * @param idx a running number
     * @param every if given, we only save if \c idx % \c every == 0.
     */
    template<class T, class R>
        void serialize_to_file_r(std::string file, const boost::shared_ptr<T> obj, const R& r, int idx=-1, int every=-1){
            namespace bar= boost::archive;
            if(every >= 0 && (idx % every != 0))
                return;
            if(idx>=0)
                file = (boost::format(file) % idx).str();
            std::cout << "serializing model to:" << file << std::endl;
            std::ofstream f(file.c_str());
            bar::binary_oarchive oa(f);
            register_objects(oa);
            r(oa);
            oa << obj;
        }

    /**
     * Same as deserialize_from_file, but with possibility of registering own polymorphic classes with the archive in a callback.
     *
     * @see deserialize_from_file
     *
     * @param file the filename. If idx is given, the file must be format string accepting an integer, e.g. "model-%04d.ser".
     * @param r a callback taking a boost archive
     * @param idx optional, if given the filename is assumed to be a format string which accepts idx
     * @return deserialized model
     */
    template<class T, class R>
        boost::shared_ptr<T> deserialize_from_file_r(std::string file, const R& r, int idx=-1){
            namespace bar= boost::archive;
            if(idx>=0)
                file = (boost::format(file) % idx).str();
            std::ifstream f(file.c_str());
            bar::binary_iarchive ia(f);
            register_objects(ia);
            r(ia);
            boost::shared_ptr<T>  obj;
            ia >> obj;
            return obj;
        }
}
#endif /* __SERIALIZATION_HELPER_HPP__ */
