#ifndef __PYTHON_HELPER_HPP__
#     define __PYTHON_HELPER_HPP__
#include <boost/shared_ptr.hpp>

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
     * Exports the classes Op, Sink, and ParameterInput, and some helpers into the main scope.
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
     * @see initialize_python export_ops export_op export_loadbatch
     *
     * @return zero iff OK
     */
    int embed_python();

    /**
     * @}
     */
}
#endif /* __PYTHON_HELPER_HPP__ */
