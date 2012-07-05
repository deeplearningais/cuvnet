#ifndef __PYTHON_HELPER_HPP__
#     define __PYTHON_HELPER_HPP__
#include <boost/shared_ptr.hpp>

namespace cuvnet
{
    class Op;
    
    /**
     * @return main scope
     */
    int initialize_python();

    int export_ops();

    int export_loadbatch(boost::function<void(unsigned int)> load_batch, boost::function<unsigned int(void)> n_batches);

    int export_op(const std::string& name, boost::shared_ptr<Op> op);

    int embed_python();
}
#endif /* __PYTHON_HELPER_HPP__ */
