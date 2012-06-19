// vim:ts=4:sw=4:et:
#ifndef __OP_UTILS_HPP__
#     define __OP_UTILS_HPP__

#include <map>
#include <fstream>
#include <cuvnet/op.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/output.hpp>

namespace cuvnet
{
    /**
     * helper class to create visitors (you can derive from this so
     * that you eg only need to implement one method).
     *
     * @ingroup op_visitors
     */
    struct op_visitor_adaptor{
        inline bool discover(Op* o)const{ return true; }
        inline void preorder(Op* o)const{ ; }
        inline void postorder(Op* o)const{ ; }
    };
    /**
     * collect all no-input ops in a list
     * @ingroup op_visitors
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
                // do not check for derivability.
                // `derivable' was added for derivative_test(), which should
                // not derive for certain parameters.
                // however, it still needs to see them to /initialize/ them.
                //if(((Input*)o)->derivable())
                plist.push_back(o);
            }
        }
    };
    /** 
     * cleanup unused data 
     * @ingroup op_visitors
     */
    struct cleanup_temp_vars_visitor : public op_visitor_adaptor{
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
            o->release_data();
            BOOST_FOREACH(Op::result_t& r, o->m_results){
                for(unsigned int i=0;i<r->result_uses.size();i++){
                    r->use(i)->get_op()->release_data();
                }
            }
        }
    };
    /**
     * collect all ops in a list in topological order
     * @ingroup op_visitors
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

            if(dynamic_cast<Sink*>(o)!=NULL) {
                // we assume the sink has been calculated in advance, so treat it
                // like an `Input' and do not bother how its value is generated
                plist.push_back(o);
                return false;
            }
            return true; // we can never reurn "false" here, since we might need this for the /forward/ pass.
        }
        inline void postorder(Op* o){
            plist.push_back(o);
        }
    };

    /**
     * determine shapes recursively
     * @ingroup op_visitors
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
     * @ingroup op_visitors
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
     * @ingroup op_visitors
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
     * @ingroup op_visitors
     */
    struct reset_needed_flags : public op_visitor_adaptor{
        std::map<Op*,bool> visited;
        inline bool discover(Op* o){
            if(dynamic_cast<Sink*>(o)!=NULL)
                return false;
            if(visited.find(o)!=visited.end()) 
                return false;
            visited[o] = true;
            return true; 
        }
        inline void preorder(Op*o){
            o->need_derivative(false);
            o->need_result(false);
            for(unsigned int i=0;i<o->get_n_params();i++)
                o->param(i)->need_derivative = false;
            for(unsigned int i=0;i<o->get_n_results();i++)
            {
                o->result(i)->need_result = false;
                o->result(i)->get_op()->need_result(false);
            }
        }
    };

    /**
     * Dump a symbolic function to a graphviz DOT file.
     *
     * very good for debugging and visualizing the models that you created.
     * @ingroup op_visitors
     */
    struct define_graphviz_node_visitor : public op_visitor_adaptor{
        std::ostream& os;
        std::vector<Op*> m_mark_order;
        Op* m_current_op;
        bool m_break_after_done;
        std::map<const void*, std::string> m_seen;
        define_graphviz_node_visitor(std::ostream& o, std::vector<Op*>* mo=NULL):os(o),m_current_op(NULL),m_break_after_done(false){
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
        inline void current_op(Op*o){m_current_op = o;}
        inline Op* current_op()const{return m_current_op;}
    };

    /**
     * Dump a symbolic function to a graphviz DOT file.
     * @ingroup op_visitors
     */
    void write_graphviz(Op& op, std::ostream& os);
    /**
     * Dump a symbolic function to a graphviz DOT file.
     * @ingroup op_visitors
     */
    void write_graphviz(Op& op, std::ostream& os, std::vector<Op*>&, Op* current=NULL);

    /**
     * @brief does a recursive forward/backward pass w.r.t. requested parameters.
     *
     * To do passes, the structure of the operator is
     * sorted topologically once (in the constructor).
     * Consecutive calles should therefore be done with
     * the same `swiper' object.
     *
     * @ingroup op_visitors
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
            :m_op(&op)
        {
                reset_needed_flags rnf;
                op.visit(rnf);

                op.result(result)->need_result = true; // this is the final res we're interested in
                op.need_result(true);                  // this is a bit redundant

                op.visit(m_topo);

                this->set_calculate_result();           // determine need_result
                op.set_calculate_derivative(paramlist); // determine need_derivative

                cleanup_temp_vars_visitor ctvv;
                op.visit(ctvv); 

                op.visit(determine_shapes_visitor());

                std::ofstream os("swiper-initial.dot");
                write_graphviz(op, os, m_topo.plist );
            }

        /**
         * determine which results need to be calculated.
         */
        void set_calculate_result(){
            BOOST_REVERSE_FOREACH(Op* op, m_topo.plist){
                BOOST_FOREACH(Op::result_t& r, op->m_results){
                    r->determine_single_results();
                    BOOST_FOREACH(boost::weak_ptr<detail::op_param<Op::value_type> >& p, r->result_uses) {
                        if(p.lock()->get_op()->need_result())
                            r->need_result = true;
                        op->need_result(true);
                    }
                }
            }
        }
        /**
         * clean up temp vars
         */
        ~swiper(){
            cleanup_temp_vars_visitor ctvv;
            m_op->visit(ctvv,true);

            reset_needed_flags rnf;
            m_op->visit(rnf); 
        }
        /**
         * does recursive forward pass on op
         */
        void fprop();
        /**
         * does recursive backward pass on op
         */
        void bprop(bool set_last_delta_to_one=true);

        /**
         * ouputs some stats of op results for debugging
         */
        void debug(unsigned int cnt, Op* o, bool results, bool params, const char* ident);
        private:
        Op* m_op;
    };

}
#endif /* __OP_UTILS_HPP__ */
