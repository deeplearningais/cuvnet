#ifndef __PYTHON_HELPER_HPP__
#     define __PYTHON_HELPER_HPP__
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>

namespace cuvnet
{
    class Op;
    
    /**
     * @addtogroup python_helpers
     * @{
     */

    /**
     * Must be called before any other python-related functions.
     *
     * @return zero iff OK
     */
    int initialize_python();

    /**
     * Exports the classes Op, Sink, and ParameterInput, and some helpers into the current scope.
     * Use this to export cuvnet to python when you compile your own python modules.
     */
    void export_module();

    /**
     * Exports the classes Op, Sink, and ParameterInput, and some helpers into the main scope.
     * This can be used to export cuvnet to python when switching to python
     * from inside the C++ program.
     *
     * @return zero iff OK
     */
    int export_ops();

    /**
     * Use this to load new data into your Op from Python.
     *
     * This is useful if you need to loop over your current dataset in python,
     * e.g. if your evaluation is written in python.
     *
     * @param load_batch a function taking only the batch number as an argument
     * @param n_batches a function returning the total number of batches
     * @return zero iff OK
     */
    int export_loadbatch(boost::function<void(unsigned int)> load_batch, boost::function<unsigned int(void)> n_batches);

    /**
     * export a specific symbolic function as a variable into the main scope.
     *
     * @param name the name in main scope
     * @param op the Op variable to be exported
     *
     * @return zero iff OK
     */
    int export_op(const std::string& name, boost::shared_ptr<Op> op);

    /**
     * start an IPython session with the current main scope.
     *
     * If not using IPython, you should have a ~/.pythonrc file which will be
     * executed. In there, you can add readline support etc. The file may also
     * be empty.
     *
     * @see initialize_python export_ops export_op export_loadbatch
     *
     * @param IPython if true, drop to IPython, not python
     *
     * @return zero iff OK
     */
    int embed_python(bool IPython=true);

    /**
     * @}
     */
}
#endif /* __PYTHON_HELPER_HPP__ */
