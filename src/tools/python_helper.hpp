#ifndef __PYTHON_HELPER_HPP__
#     define __PYTHON_HELPER_HPP__
#include <boost/python.hpp>

namespace cuvnet
{
    
    /**
     * @return main scope
     */
    boost::python::object initialize_python();

    void export_ops();

    void embed_python();
}
#endif /* __PYTHON_HELPER_HPP__ */
