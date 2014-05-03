// vim:ts=4:sw=4:et:
#ifndef __OP_UTILS_HPP__
#     define __OP_UTILS_HPP__

#include <map>
#include <fstream>
#include <cuvnet/op.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/convolve.hpp>
#include <algorithm>
#include <log4cxx/logger.h>

namespace cuvnet
{

    /**
     * helper class to create visitors (you can derive from this so
     * that you eg only need to implement one method).
     *
     * @ingroup op_visitors
     */
    struct op_visitor_adaptor{
        /**
         * called when a new node is discovered.
         * @param o the discovered node
         * @return true iff the node should be processed.
         */
        inline bool discover(Op* o)const{ return true; }
        /**
         * process a node before its children.
         * @param o the node
         */
        inline void preorder(Op* o)const{ ; }
        /**
         * process a node after its children.
         * @param o the node
         */
        inline void postorder(Op* o)const{ ; }
    };

    /**
     * helper class to create visitors (you can derive from this so
     * that you eg only need to implement one method).
     *
     * This one only visits every op once
     *
     * @ingroup op_visitors
     */
    struct op_visitor_once_adaptor{
        mutable std::map<Op*,bool> visited; ///< keeps track of nodes we've seen
        bool m_visit_sinks; ///< if true, walk past sinks
        /**
         * ctor.
         * @param visit_sinks if false, do not walk past sinks.
         */
        op_visitor_once_adaptor(bool visit_sinks=false)
            :m_visit_sinks(visit_sinks)
        {
        }

        void forget_visited(){
            visited.clear();
        }

        /// @overload
        inline bool discover(Op* o)const{
            // do not continue past Sinks---they are treated as Inputs!
            if(!m_visit_sinks && dynamic_cast<Sink*>(o)!=NULL)
                return false;
            if(visited.find(o)!=visited.end())
                return false;
            visited[o]=true;
            return true;
        }
        /**
         * does nothing.
         * @overload
         */
        inline void preorder(Op* o)const{ ; }
        /**
         * does nothing.
         * @overload
         */
        inline void postorder(Op* o)const{ ; }
    };

    /**
     * collect all ParameterInput ops in a list.
     *
     * This is mainly used in two cases:
     * - in testing to figure out for which objects to call a derivative test,
     * - from python, to identify ops by name or memory address.
     *
     * @ingroup op_visitors
     */
    struct param_collector_visitor
        : public op_visitor_once_adaptor{
        typedef std::vector<Op*> container_type; ///< type of container which stores found objects
        container_type     plist; ///< container which stores found objects
        std::string m_name_query; ///< a string query (optional)
        const Op*          m_ptr_query; ///< a pointer query (optional)
        /**
         * collect everything that is a ParameterInput.
         */
        param_collector_visitor(): op_visitor_once_adaptor(true), m_ptr_query(NULL){}
        /**
         * filter by name (must match exactly).
         * @param name the name to filter for.
         */
        param_collector_visitor(const std::string& name)
        :op_visitor_once_adaptor(true), m_name_query(name), m_ptr_query(NULL){
        }
        /**
         * filter by pointer.
         * @param op the memory address to filter for.
         */
        param_collector_visitor(const Op* op)
        :op_visitor_once_adaptor(true),m_ptr_query(op){
        }
        /**
         * @overload
         */
        inline void preorder(Op* o){
            // do not check for derivability.
            // `derivable' was added for derivative_test(), which should
            // not derive for certain parameters.
            // however, it still needs to see them to /initialize/ them.
            //if(((Input*)o)->derivable())
            //std::cout << "test: "<<boost::lexical_cast<std::string>(o)<<std::endl;
            //std::cout << "  m_name_query.length():" << m_name_query.length() << std::endl;
            //std::cout << "  m_ptr_query:" << boost::lexical_cast<std::string>(m_ptr_query) << std::endl;

            if(!m_name_query.length() && m_ptr_query == NULL){
                ParameterInput* pi = dynamic_cast<ParameterInput*>(o);
                if(pi)
                    plist.push_back(pi);
            }
            else if(m_name_query.length())
            {
                ParameterInput* pi = dynamic_cast<ParameterInput*>(o);
                if(pi && m_name_query == pi->name())
                    plist.push_back(pi);

                Sink* si = dynamic_cast<Sink*>(o);
                if(si && m_name_query == si->name())
                    plist.push_back(si);
            }
            else if(m_ptr_query && m_ptr_query == o)
                plist.push_back(o);
            //else std::cout << "  --> no match: "<<boost::lexical_cast<std::string>(o)<<std::endl;
        }
    };

    /**
     * find a parameter by name.
     *
     * @param f the function object to search in
     * @param name the name to look for (must be unique)
     * @return the found object
     * @throw runtime_error if not found
     *
     * @ingroup op_visitors
     */
    template<class T>
    inline T* get_node(const boost::shared_ptr<Op>& f, const std::string& name){
        param_collector_visitor pcv(name);
        f->visit(pcv,true);
        if(pcv.plist.size()==0)
            throw std::runtime_error("Could not find node named `"+name+"'");
        if(pcv.plist.size() > 1)
            throw std::runtime_error("Multiple matches for node named `"+name+"'");
        return dynamic_cast<T*>(pcv.plist.front());
    }

    /**
     * find a parameter by address>
     *
     * @param f the function object to search in
     * @param query the memory address to look for
     * @return the found object
     * @throw runtime_error if not found
     * @ingroup op_visitors
     */
    template<class T>
    inline T* get_node(const boost::shared_ptr<Op>& f, Op* query){
        param_collector_visitor pcv(query);
        f->visit(pcv,true);
        if(pcv.plist.size()==0)
            throw std::runtime_error("Could not find parameter with address `"+boost::lexical_cast<std::string>(query)+"'");
        return dynamic_cast<T*>(pcv.plist.front());
    }


    /**
     * cleanup unused data.
     * @ingroup op_visitors
     */
    struct cleanup_temp_vars_visitor
        : public op_visitor_once_adaptor{

        /// @overload
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
     * determine which nodes need to be evaluated in fprop and in bprop.
     */
    struct determine_exec_order{
        /// the container type used to store the found objects.
        typedef std::vector<Op*> container_type;

        /**
         * initialize the ordering.
         *
         * @param o the op we want to derive for
         * @param results other, side results we want to calculate
         * @param parameters parameters that we want to derive for
         */
        void init(Op* o, int o_res,
                const std::vector<std::pair<Op*, int> >& results,
                const std::vector<Op*>& parameters);

        /// the container used to store the nodes that need to be executed during fprop in topological order.
        container_type     fprop_nodelist;

        /// the container used to store the nodes that need to be executed during bprop in topological order.
        container_type     bprop_nodelist;

        void determine_bprop_list(Op* o, const container_type& parameters, const container_type& plainfprop);

        void determine_fprop_list(Op* o, int o_res, const std::vector<std::pair<Op*, int> >& results);

        private:
        bool find_in_results(Op* query,
                const std::vector<std::pair<Op*, int> >& other_queries,
                Op* search_start);
    };

    /**
     * collect all ops in a list in topological order.
     * @ingroup op_visitors
     */
    struct toposort_visitor : public op_visitor_adaptor{
        /// the container type used to store the found objects
        typedef std::vector<Op*> container_type;
        /// the container used to store the found objects
        container_type     plist;
        /// keep track of what has been visited
        std::map<Op*,bool> visited;
        /// default ctor.
        toposort_visitor(){}
        /// @overload
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
            return true; // we can never return "false" here, since we might need this for the /forward/ pass.
        }
        /// @overload
        inline void postorder(Op* o){
            plist.push_back(o);
        }
        /// forget everything
        inline void clear(){
            plist.clear();
            visited.clear();
        }
    };

    /**
     * determine shapes recursively.
     * @ingroup op_visitors
     */
    struct determine_shapes_visitor :public op_visitor_once_adaptor{
        /**
         * here we mark all parameters which have been updated.
         * only if all parameters have been updated, we can update the result.
         */
        std::map<detail::op_param<Op::value_type>*, bool> m_marked_params;

        /// as long as this is false after a run, some result shapes are unknown.
        bool done;

        /// ctor.
        determine_shapes_visitor() : op_visitor_once_adaptor(true), done(false){} // also visit sinks!

        /// @overload
        inline void postorder(Op* o){
            if(dynamic_cast<DeltaSink*>(o)) // delta sinks do not fprop, just take what they get for bprop.
                return;
            BOOST_FOREACH(Op::param_t& p, o->m_params){
                if(m_marked_params.find(p.get()) == m_marked_params.end()){
                    done = false;
                    return; // cannot initialize this op (yet)!
                }
            }

            o->_determine_shapes();
            // push from result to result-users
            BOOST_FOREACH(Op::result_t& r, o->m_results){
                for(unsigned int i=0;i<r->result_uses.size();i++){
                    auto u = r->use(i);
                    u->shape = r->shape;
                    m_marked_params[u.get()] = true;
                }
            }
        }
    };

    /**
     * determine shapes using the determine_shapes_visitor.
     * The visitor cannot be applied directly, since the purely recursive
     * approach (which includes looking at results of ops, not only parameters)
     * does not result in a topographic ordering of the complete graph.
     *
     * @todo write a topological sort that includes /all/ connected sinks
     */
    inline void determine_shapes(Op& op, bool results_too=true){
        determine_shapes_visitor dsv;
        while(!dsv.done){
            dsv.done = true;
            dsv.forget_visited();
            op.visit(dsv, results_too);
        }
    }

    /**
     * reset the `delta_set' flag before a bprop-pass.
     * @ingroup op_visitors
     */
    struct reset_delta_set_flag : public op_visitor_adaptor{
        /// @overload
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
        /// @overload
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
        std::map<Op*,bool> visited; ///< keeps track of Ops that we have seen
        /// @overload
        inline bool discover(Op* o){
            // do not continue past Sinks---they are treated as Inputs!
            if(dynamic_cast<Sink*>(o)!=NULL)
                return false;
            if(visited.find(o)!=visited.end())
                return false;
            visited[o] = true;
            return true;
        }
        /// @overload
        inline void preorder(Op*o){
            o->need_derivative(false);
            o->need_result(false);
            for(unsigned int i=0;i<o->get_n_params();i++)
                o->param(i)->need_derivative = false;
            for(unsigned int i=0;i<o->get_n_results();i++)
            {
                boost::shared_ptr<detail::op_result<Op::value_type> > res
                    = o->result(i);
                res->need_result = false;
                res->get_op()->need_result(false);

                // HACK: reference counting of functionoutputs is wrong if the
                // result points back to them.
                // i.e. loss.use_count() is 2, and therefore it will not be
                // destroyed, if loss.result points back to loss.
                if(res->result_uses.size() == 0)
                    res->op.reset();
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
        bool m_verbose;
        std::vector<Op*> m_fmark_order; ///< eg the topological ordering of the graph (fprop)
        std::vector<Op*> m_bmark_order; ///< eg the topological ordering of the graph (bprop)
        Op* m_current_op; ///< for stepping through a function, set this to the current position
        bool m_break_after_done; ///< for debugging: is set to true when NaN is found, so that learning can be stopped
        std::map<const void*, std::string> m_seen; ///< records stuff we've seen already, so we do not define nodes twice
        std::string m_group_filter; ///< if non-empty, only print nodes with this group
        std::ostringstream m_node_defs; ///< contains all node definitions
        std::ostringstream m_edge_defs; ///< contains all edge definitions
        /**
         * ctor.
         * @param fmo if given, contain the topological ordering in the graph for fprop. Indices will be appended to nodes in this list.
         * @param bmo if given, contain the topological ordering in the graph for bprop. Indices will be appended to nodes in this list.
         */
        define_graphviz_node_visitor(bool verbose, std::vector<Op*>* fmo=NULL, std::vector<Op*>* bmo=NULL, std::string group_filter=""):m_verbose(verbose),m_current_op(NULL),m_break_after_done(false),m_group_filter(group_filter){
            if(fmo)
                m_fmark_order = *fmo;
            if(bmo)
                m_bmark_order = *bmo;
        }
        /**
         * @overload
         */
        inline bool discover(Op* o){
            if(m_seen.find(o)!=m_seen.end()) return false;
            return true;
        }
        /// @overload
        void preorder(Op*o);
        /// @overload
        void postorder(Op*o);

        /// internal.
        std::string define_data_ptr(const Op::value_ptr& vp);
        /// Useful for debugging by stepping: set the current op, which is marked in red.
        inline void current_op(Op*o){m_current_op = o;}
        /// @return the current op to be marked.
        inline Op* current_op()const{return m_current_op;}
    };

    /**
     * Dump a symbolic function to a graphviz DOT file.
     * @param op the Op to be visualized
     * @param os the stream to write to
     * @ingroup op_visitors
     */
    void write_graphviz(Op& op, std::ostream& os, bool verbose);
    /**
     * Dump a symbolic function to a graphviz DOT file.
     * @param op the Op to be dumped
     * @param os the stream to write to
     * @param fmo the mark order (topological sorting) of op for fprop
     * @param bmo the mark order (topological sorting) of op for bprop
     * @param current the current node, to be marked (useful for stepping through function evaluation).
     * @ingroup op_visitors
     */
    void write_graphviz(Op& op, std::ostream& os, bool verbose, std::vector<Op*>& fmo, std::vector<Op*>& bmo, Op* current=NULL);

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
        determine_exec_order m_topo; ///< contains exec order information for fprop/bprop
        int m_verbosity;

        /**
         * Other functions that are part of the Op graph and need to
         * be evaluated, but not derived for.
         */
        std::vector<std::pair<Op*,int> > m_other_funcs;

        /**
         * turn cleaning up temp vars off/on
         */
        inline void set_cleanup_temp_vars(bool b){
            m_cleanup_temp_vars = b;
        }

        /**
         * change verbosity of dot output.
         * @param verbosity if greater 0, dumps dot file, if greater 1, dump shapes and execution order as well
         */
        inline void set_verbosity(int verbosity){
            m_verbosity = verbosity;
        }

        /**
         * constructor
         *
         * @param op      the operator to do swipes on
         * @param result  the result of the operator to optimize
         * @param paramlist the list of parameters w.r.t. which do swipes
         * @param verbosity if greater 0, dumps dot file, if greater 1, dump shapes and execution order as well
         */
        swiper(Op& op, int result, const param_collector_visitor::container_type& paramlist, int verbosity=0)
            : m_verbosity(verbosity),
            m_op(&op),
            m_result(result),
            m_paramlist(paramlist),
            m_cleanup_temp_vars(true)
        {
                init();
        }

        /**
         * add a function that needs to be evaluated, but not derived
         * for, and is part of the main Op's graph.
         *
         * @param op the other operator we're interested in
         * @param result the number of the result of the op we're interested in
         * @param call_init if true, call init, otherwise you have to call it yourself so that changes take effect.
         *                  When adding multiple 'other' results, this can avoide multiple useless init() calls.
         */
        void request_other_result(Op& op, int result=0, bool call_init=true);

        /**
         * calls write_graphviz with the current topological order.
         *
         * @param filename where to store the dot file
         */
        void dump(const std::string& filename, bool verbose){
            std::ofstream os(filename);
            write_graphviz(*m_op, os, verbose, m_topo.fprop_nodelist, m_topo.bprop_nodelist );
        }

        /**
         * is called by the constructor and needs to be called again if the Op
         * from the constructor was modified.
         */
        void init();

        /**
         * determine which results need to be calculated.
         */
        void set_calculate_result();

        /**
         * clean up temp vars.
         */
        ~swiper();
        /**
         * does recursive forward pass on op.
         */
        void fprop();
        /**
         * does recursive backward pass on op.
         * @param set_last_delta_to_one if true, backpropagates (tensor of)
         * ones. Otherwise, assumes that the user has set a different value in
         * delta.
         */
        void bprop(bool set_last_delta_to_one=true);

        /**
         * used in stepping through functions; writes a DOT file containing infos about current evaluation state.
         * @param cnt a running number, increasing with every node in the graph
         * @param o the current node in the graph
         * @param results internal
         * @param params internal
         * @param ident a string to be used in the filename
         */
        void debug(unsigned int cnt, Op* o, bool results, bool params, const char* ident);


        /**
         * checks if all the parameters supplied in the constructor are part of the function graph.
         * @throw runtime_error if some parameters are not part of the function graph.
         */
        void check_param_existence()const;

        private:
        Op* m_op;
        int m_result;
        param_collector_visitor::container_type m_paramlist;

        /**
         * If this is set to false, no cleaning up of values stored in sinks
         * etc. is done when the swiper is destroyed.
         */
        bool m_cleanup_temp_vars;

    };

    /**
     * exception thrown by valid_shape_info when finished.
     */
    struct valid_shape_info_finished{};

    /**
     * Determine how to transform the teacher to a a series of convolutions/pooling
     * operations given only valid convolutions.
     *
     * - Valid convolutions decrease the size of the input.
     *   A teacher containing pixel-wise annotation should therefore be cropped
     *   by a margin.
     * - Pooling operations, on the other hand, change the size by a factor.
     *   A pixel-wise annotated teacher should therefore be /scaled/.
     *
     * This function traverses the path from the teacher to the input and can be
     * used to retrieve the combined scaling/cropping operations.
     *
     * After running, the object has a scale and a crop variable set for each
     * dimension. The original image is related to the output as follows:
     *
     * output_size = (size - crop)/scale
     *
     * where cropping should take away crop/2 from both sides of the input.
     *
     * @note for convenience, this is currently implemented using depth-first
     * search. Thus, we cannot guarantee /shortest/ path or efficiency,
     * however, this should not be a problem since most graphs will be small
     * and the function will only be called once.
     *
     * @ingroup op_visitors
     */
    struct valid_shape_info
        : public op_visitor_once_adaptor{
        typedef std::vector<Op*> container_type;
        Op *m_begin;  ///< start point of evaluation (input images)
        Op *m_end;    ///< end point of evaluation (output images)

        /**
         * ctor.
         * @param begin start point of search (input images)
         * @param end   end point of search (output images)
         */
        valid_shape_info(boost::shared_ptr<Op> begin, boost::shared_ptr<Op> end)
            :m_begin(begin.get()), m_end(end.get())
        {
            log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("op"));
            if(!begin || !end){
                LOG4CXX_WARN(log, "valid_shape_info: cannot determine path due to NULL-argument!");
                return;
            }
            bool ok = false;
            try{
                end->visit(*this);
            }catch(valid_shape_info_finished){
                determine_shapes();
                ok = true;
            }
            if(!ok){
                LOG4CXX_FATAL(log, "valid_shape_info: no path found!");
            }
        }

        container_type     plist; ///< stores the nodes on the path between begin and end
        std::map<Op*,bool> visited; ///< keeps track of what has been visited
        /// @overload
        inline bool discover(Op* o){
            if(visited.find(o)!=visited.end())
                return false;
            visited[o] = true;

            if(o == m_begin) {
                plist.push_back(o);
                throw valid_shape_info_finished();
            }
            return true;
        }
        /// @overload
        inline void preorder(Op* o){
            plist.push_back(o);
        }
        /// @overload
        inline void postorder(Op* o){
            plist.pop_back();
        }

        float o2i_scale;
        float i_margin_l, i_margin_r;

        /// traverse plist to determine what the cropping and scaling parameters should be.
        void determine_shapes();

        std::pair<float, float> o2i(float y, float x)const;
        std::pair<float, float> i2o(float y, float x)const;
    };
}
#endif /* __OP_UTILS_HPP__ */
