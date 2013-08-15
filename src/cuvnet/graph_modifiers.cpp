#include "graph_modifiers.hpp"
#include <cuvnet/op_utils.hpp>

namespace cuvnet { namespace graph_modifiers {
    substitute_op_with_input::substitute_op_with_input(const op_ptr& p)
        :m_original_op(p)
    {
        // determine the shape of p's output(s)
        determine_shapes(*p);

        // create an input object with the same shape
        cuvAssert(p->get_n_results() == 1);

        boost::shared_ptr<detail::op_result<matrix> >  r = p->result();
        m_input = boost::make_shared<ParameterInput>(r->shape, "substituted input");
        boost::shared_ptr<detail::op_result<matrix> >  ir = m_input->result();
        // we first determine all uses of p's (only) result
        while(r->result_uses.size()){
            boost::shared_ptr<detail::op_param<matrix> >  u 
                = r->use(r->result_uses.size()-1);

            // remove forward pointer from p's result to param of following op
            r->remove(u.get(), false);
            u->remove(r.get());
            
            // tell input that it is being used by u now
            ir->result_uses.push_back(u);
            u->param_uses.push_back(ir);

        }
    }

    substitute_op_with_input::~substitute_op_with_input(){
        boost::shared_ptr<detail::op_result<matrix> >  ir = m_input->result();
        boost::shared_ptr<detail::op_result<matrix> >  r = m_original_op->result();

        // we first determine all uses of m_input's (only) result
        while(ir->result_uses.size()){
            boost::shared_ptr<detail::op_param<matrix> >  u 
                = ir->use(ir->result_uses.size()-1);

            u->substitute(ir, r);
            
            r->result_uses.push_back(u);
            ir->remove(u.get());
        }
        m_input.reset();
    }
} }
