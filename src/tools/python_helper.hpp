#ifndef __PYTHON_HELPER_HPP__
#     define __PYTHON_HELPER_HPP__
#include <boost/python.hpp>

namespace cuvnet
{
    
    void initialize_python();

    void export_ops();

    void embed_python();
}
#endif /* __PYTHON_HELPER_HPP__ */
