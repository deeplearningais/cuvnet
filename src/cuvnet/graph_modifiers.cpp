#include "graph_modifiers.hpp"
#include <cuvnet/op_utils.hpp>

namespace cuvnet { namespace graph_modifiers {
    substitute_op_with_input::substitute_op_with_input(boost::shared_ptr<Op> p)
        :m_original_op(p)
    {
        // determine the shape of p's output(s)
        determine_shapes(*p);

        // create an input object with the same shape
        cuvAssert(p->get_n_results() == 1);
        m_input = boost::make_shared<Input>(p->result()->shape, "substituted input");

        // we first determine all uses of p's (only) result
        for(unsigned int ru = 0; ru < p->result()->result_uses.size(); ru++){
            boost::shared_ptr<detail::op_param<matrix> >  u = p->result()->use(ru);

            // swap backwards pointer from param of following op to p's result
            u->substitute(p->result(), m_input->result());
            
            // tell input that it is being used by u now
            m_input->result()->result_uses.push_back(u);

            // remove forward pointer from p's result to param of following op
            p->result()->remove(u.get());

        }
    }

    substitute_op_with_input::~substitute_op_with_input(){
        // we first determine all uses of m_input's (only) result
        for(unsigned int ru = 0; ru < m_input->result()->result_uses.size(); ru++){
            boost::shared_ptr<detail::op_param<matrix> >  u = m_input->result()->use(ru);

            u->substitute(m_input->result(), m_original_op->result());
            
            m_original_op->result()->result_uses.push_back(u);

            m_input->result()->remove(u.get());
        }
        m_input.reset();
    }
} }
