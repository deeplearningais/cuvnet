// vim:ts=4:sw=4:et:
#ifndef __OP_UTILS_HPP__
#     define __OP_UTILS_HPP__

#include <map>
#include <fstream>
#include <cuvnet/op.hpp>
#include <cuvnet/ops/input.hpp>

namespace cuvnet
{
    /**
     * helper class to create visitors (you can derive from this so
     * that you e.g. only need to implement one method)
     */
    struct op_visitor_adaptor{
        inline bool discover(Op* o)const{ return true; }
        inline void preorder(Op* o)const{ ; }
        inline void postorder(Op* o)const{ ; }
    };
    /**
     * collect all no-input ops in a list
     */
    struct param_collector_visitor : public op_visitor_adaptor{
        typedef std::vector<Op*> container_type;
        container_type     plist;
        std::map<Op*,bool> visited;
        inline bool discover(Op* o){
            if(visited.find(o)!=visited.end())
                return false;
            visited[o]=true;
            return true;
        }
        inline void preorder(Op* o){
            if(o->get_n_params()==0)
            {
                if(((Input*)o)->derivable())
                    plist.push_back(o);
            }
        }
    };
    /**
     * collect all ops in a list in topological order
     */
    struct toposort_visitor : public op_visitor_adaptor{
        typedef std::vector<Op*> container_type;
        container_type     plist;
        std::map<Op*,bool> visited;
        toposort_visitor(){}
        inline bool discover(Op* o){
            if(visited.find(o)!=visited.end()) 
                return false;
            visited[o] = true;
            return true; // we can never reurn "false" here, since we might need this for the /forward/ pass.
        }
        inline void postorder(Op* o){
            plist.push_back(o);
        }
    };

    /**
     * determine shapes recursively
     */
    struct determine_shapes_visitor :public op_visitor_adaptor{
        determine_shapes_visitor(){}
        inline void postorder(Op* o)const{
            o->_determine_shapes();
            // push from result to result-users
            BOOST_FOREACH(Op::result_t& r, o->m_results){
                for(unsigned int i=0;i<r->result_uses.size();i++){
                    r->use(i)->shape = r->shape;
                }
            }
        }
    };

    /**
     * reset the `delta_set' flag before a bprop-pass
     */
    struct reset_delta_set_flag : public op_visitor_adaptor{
        inline void preorder(Op*o)const{
            BOOST_FOREACH(Op::result_t& r, o->m_results){
                r->delta_set = false;
            }
            // TODO: is this necessary? ::
            BOOST_FOREACH(Op::param_t& r, o->m_params){
                // we may have written a delta to a followup op which is not
                // part of the hierarchy!
                for(unsigned int i=0;i<r->param_uses.size();i++)
                    r->use(i)->delta_set = false;
            }
        }
    };
    /**
     * reset the `value_set' flag before a fprop-pass
     */
    struct reset_value_set_flag : public op_visitor_adaptor{
        inline void preorder(Op*o)const{
            BOOST_FOREACH(Op::param_t& r, o->m_params){
                r->value_set = false;
            }
            // TODO: is this necessary? ::
            BOOST_FOREACH(Op::result_t& r, o->m_results){
                // we may have written a result to a followup op which is not
                // part of the hierarchy!
                for(unsigned int i=0;i<r->result_uses.size();i++)
                    r->use(i)->value_set = false;
            }
        }
    };
    /**
     * set need_derivative and need_result to false
     */
    struct reset_needed_flags : public op_visitor_adaptor{
        inline void preorder(Op*o)const{
            for(unsigned int i=0;i<o->get_n_params();i++)
                o->param(i)->need_derivative = false;
            for(unsigned int i=0;i<o->get_n_results();i++)
                o->result(i)->need_result = false;
        }
    };

    struct define_graphviz_node_visitor : public op_visitor_adaptor{
        std::ostream& os;
        std::vector<Op*> m_mark_order;
        std::map<const void*, std::string> m_seen;
        define_graphviz_node_visitor(std::ostream& o, std::vector<Op*>* mo=NULL):os(o){
            if(mo)
                m_mark_order = *mo;
        }
        inline bool discover(Op* o){
            if(m_seen.find(o)!=m_seen.end()) return false;
            return true;
        }
        void preorder(Op*o);
        void postorder(Op*o);
        std::string define_data_ptr(const Op::value_ptr& vp);
    };
    void write_graphviz(Op& op, std::ostream& os);
    void write_graphviz(Op& op, std::ostream& os, std::vector<Op*>&);

    /**
     * does a recursive forward/backward pass w.r.t. 
     * requested parameters.
     *
     * To do passes, the structure of the operator is
     * sorted topologically once (in the constructor).
     * Consecutive calles should therefore be done with
     * the same `swiper' object.
     */
    struct swiper{
        toposort_visitor m_topo;
        /**
         * constructor
         *
         * @param op      the operator to do swipes on
         * @param result  the result of the operator to optimize
         * @param paramlist the list of parameters w.r.t. which do swipes
         */
        swiper(Op& op, int result, const param_collector_visitor::container_type& paramlist)
        {
                op.visit(reset_needed_flags());
                op.result(result)->need_result = true;
                op.set_calculate_derivative(paramlist);
                op.visit(m_topo);
                op.visit(determine_shapes_visitor());
                std::ofstream os("swiper-initial.dot");
                write_graphviz(op, os, m_topo.plist );
            }
        /**
         * does recursive forward pass on op
         */
        void fprop();
        /**
         * does recursive backward pass on op
         */
        void bprop(bool set_last_delta_to_one=true);
    };

}
#endif /* __OP_UTILS_HPP__ */
