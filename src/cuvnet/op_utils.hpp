// vim:ts=4:sw=4:et:
#ifndef __OP_UTILS_HPP__
#     define __OP_UTILS_HPP__

#include <cuvnet/op.hpp>

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
                plist.push_back(o);
        }
    };
    /**
     * collect all ops in a list in topological order
     */
    struct toposort_visitor : public op_visitor_adaptor{
        typedef std::vector<Op*> container_type;
        container_type     plist;
        std::map<Op*,bool> visited;
        bool               deriv_only;
        toposort_visitor(bool deriv):deriv_only(deriv){}
        inline bool discover(Op* o){
            if(visited.find(o)!=visited.end()) return false;
            if(deriv_only){
                if(o->m_params.size()==0){// input
                    visited[o] = true;
                    return true;
                }
                for (int i = 0; i < o->m_params.size(); ++i)
                {
                    // at least one parameter should have this set
                    if(o->m_params[i]->need_derivative){
                        visited[o] = true;
                        return true;
                    }
                }
            }
            return true;
        }
        inline void postorder(Op* o){
            plist.push_back(o);
        }
    };

    /**
     * determine shapes recursively
     */
    struct determine_shapes_visitor :public op_visitor_adaptor{
        bool deriv_only;
        determine_shapes_visitor(bool deriv=false):deriv_only(deriv){}
        inline void postorder(Op* o)const{
            o->_determine_shapes();
            // push from result to result-users
            BOOST_FOREACH(Op::result_t& r, o->m_results){
                for(unsigned int i=0;i<r->result_uses.size();i++){
                    if(!deriv_only || r->use(i)->need_derivative)
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
        }
    };

    struct define_graphviz_node_visitor : public op_visitor_adaptor{
        std::ostream& os;
        std::vector<Op*> m_mark_order;
        define_graphviz_node_visitor(std::ostream& o, std::vector<Op*>* mo=NULL):os(o){
            if(mo)
                m_mark_order = *mo;
        }
        void preorder(Op*o);
        void postorder(Op*o);
    };

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
         * @param deriv   whether to only do passes w.r.t. the named parameters
         * @param paramlist the list of parameters w.r.t. which do swipes
         */
        swiper(Op& op, bool deriv, const param_collector_visitor::container_type& paramlist)
            :m_topo(deriv){
                op.set_calculate_derivative(paramlist);
                op.visit(m_topo);
                op.visit(determine_shapes_visitor());
            }
        /**
         * does recursive forward pass on op
         */
        void fprop();
        /**
         * does recursive backward pass on op
         */
        void bprop();
    };

    void write_graphviz(Op& op, std::ostream& os);
    void write_graphviz(Op& op, std::ostream& os, std::vector<Op*>&);
}
#endif /* __OP_UTILS_HPP__ */
