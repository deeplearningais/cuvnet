#include <cuvnet/op.hpp>
#include <cuvnet/ops/input.hpp>

namespace cuvnet
{
    namespace graph_modifiers
    {
        /**
         * removes an op from the graph and substitutes it with an input op of
         * the same shape. The operation is undone when the object is destroyed.
         */
        struct substitute_op_with_input{
            typedef boost::shared_ptr<Op>               op_ptr;
            typedef boost::shared_ptr<ParameterInput>   input_ptr;

            op_ptr m_original_op;
            input_ptr m_input;

            substitute_op_with_input(const op_ptr& p);
            ~substitute_op_with_input();
        };
    }
}
