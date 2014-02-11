#include <fstream>
#include <ext/functional>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include "op_utils.hpp"
#include <cuvnet/ops/output.hpp>


using namespace cuvnet;
template<class ArgumentType, class ResultType>
struct ptr_caster
: public std::unary_function<ArgumentType*, ResultType*>
{
    ResultType* operator()(ArgumentType*s)const{ return (ResultType*)s; }
};

std::string define_graphviz_node_visitor::define_data_ptr(const Op::value_ptr& p){
    if(!p)
        return "";
    if(m_seen.find(&p.cdata())!=m_seen.end())
        return m_seen[&p.cdata()];
    m_seen[&p.cdata()] = "v"+boost::lexical_cast<std::string>(&p.cdata());
    m_node_defs << m_seen[&p.cdata()] << " [ label=\"(";
    std::vector<unsigned int> shape = p.cdata().shape();
    std::copy(shape.begin(),shape.end(),std::ostream_iterator<unsigned int>(m_node_defs,","));
    m_node_defs  << ")\" ] ; "<<std::endl;
    return m_seen[&p.cdata()];
}
void define_graphviz_node_visitor::preorder(Op* o){

	// fill in defaults
	detail::graphviz_node n;
	n.shape = "record";
	n.color = "black";
    if(!o->need_result())
        n.fillcolor = "white";
    else if(!o->need_derivative())
        n.fillcolor = "gray90";
    else
        n.fillcolor = "gray70";
	n.style = "filled";
    bool is_input = dynamic_cast<ParameterInput*>(o);
    bool is_sink  = dynamic_cast<Sink*>(o);
    Op* sink_prev = NULL;
    if(is_input){
        if(!o->need_result())
            n.fillcolor = "lemonchiffon1";
        else if(!o->need_derivative())
            n.fillcolor = "goldenrod1";
        else
            n.fillcolor = "goldenrod3";

        Op::value_ptr p = ((ParameterInput*)o)->data_ptr();
        if(!p);
        else{
            std::ostringstream ss;
            ss<<" (";
            for(int i=0;i<p.cdata().ndim();i++)
                ss<<p.cdata().shape(i)<<",";
            ss<<")";
            n.label += ss.str();
        }
    }else if(is_sink){
        sink_prev = o->param(0)->use(0)->get_op().get();
        if(sink_prev->need_result())
            n.fillcolor = "cadetblue";
        else
            n.fillcolor = "lightcyan";
    }
	o->_graphviz_node_desc(n);
    if(m_verbose)
    n.label += " } | {" + boost::lexical_cast<std::string>(o);
    if(o->get_label().size())
    {
        n.label = n.label + " } | { "+ o->get_label();
        n.penwidth = 4.f;
    }
    if(o->get_group().size()){
        n.group = o->get_group();
    }

	if(m_fmark_order.size()){
		std::vector<Op*>::iterator fit = std::find(m_fmark_order.begin(),m_fmark_order.end(),o);
		std::vector<Op*>::iterator bit = std::find(m_bmark_order.begin(),m_bmark_order.end(),o);
        if(m_verbose){
		if(fit!=m_fmark_order.end() && bit!=m_bmark_order.end())
			n.label += " [" + boost::lexical_cast<std::string>(std::distance(m_fmark_order.begin(),fit))
			+ ", " + boost::lexical_cast<std::string>(std::distance(m_bmark_order.begin(),bit))+"]";
        else if(fit!=m_fmark_order.end())
			n.label += " [" + boost::lexical_cast<std::string>(std::distance(m_fmark_order.begin(),fit))+",]";
        else if(bit!=m_bmark_order.end())
			n.label += " [," + boost::lexical_cast<std::string>(std::distance(m_bmark_order.begin(),bit))+"]";
        }
	}
    if(current_op()==o){
        n.fillcolor = "firebrick1";
    }

    std::string type;
    if(is_input)
        type = "input ";
    else if(is_sink)
        type = "sink ";
    else 
        type = "generic ";

    // define the op-node itself
    std::string opstr = "n" + boost::lexical_cast<std::string>( o );
    std::string groupstr = n.group.size() ? "group=\"" + n.group + "\"," : "";
    std::string paramsstr, resultsstr;
	BOOST_FOREACH(Op::param_t& p, o->m_params){
        paramsstr = paramsstr + "<p" + boost::lexical_cast<std::string>(p->param_number) + "> " + boost::lexical_cast<std::string>(p->param_number);
        if(m_verbose){
            std::string shape;
            if(p->shape.size()){
                shape = " (";
                for(unsigned int i=0;i<p->shape.size();i++){
                    shape += boost::lexical_cast<std::string>(p->shape[i]) + ((i==p->shape.size()-1) ? " " : ", ");
                }
                shape += ")";
            }
            paramsstr += shape;
        }
        if(o->get_n_params()-1 != p->param_number)
            paramsstr = paramsstr + " | ";
    }
    if(paramsstr.size() != 0)
        paramsstr = "{ " + paramsstr + " } |";

	BOOST_FOREACH(Op::result_t& r, o->m_results){
        resultsstr = resultsstr + "<r" + boost::lexical_cast<std::string>(r->result_number) + "> " + boost::lexical_cast<std::string>(r->result_number);

        std::string shape;
        if(r->shape.size()){
            shape = " (";
            for(unsigned int i=0;i<r->shape.size();i++){
                shape += boost::lexical_cast<std::string>(r->shape[i]) + ((i==r->shape.size()-1) ? " " : ", ");
            }
            shape += ")";
        }
        resultsstr += shape;

        if(o->get_n_results()-1 != r->result_number)
            resultsstr = resultsstr + " | ";
    }
    if(resultsstr.size() != 0)
        resultsstr = " | { " + resultsstr + " }";

    if(!m_verbose && o->get_n_results() == 1){
        resultsstr = "";
        n.label = "<r0> " + n.label;
    }
    m_seen[o] = opstr;
    if(m_group_filter.size()             // we're filtering
            && m_group_filter != "__empty"  // for non-empty elements
            && o->get_group() != m_group_filter) // and do not match
        return;
    if(m_group_filter == "__empty"  // we're filtering for empty elements
            && o->get_group() != "") // and the element is not empty
        return;
    //if(!o->need_result())
    //    return;
	m_node_defs << opstr
	   << " ["
	   << " URL=\""<<type<<boost::lexical_cast<std::string>(o)<<"\","
	   << " label=\"{" << paramsstr << "{ "<<n.label<<" } " <<  resultsstr << " }\","
	   << groupstr
	   << " shape=\""<<n.shape<<"\","
	   << " style=\""<<n.style<<"\","
	   << " penwidth=\""<<n.penwidth<<"\","
	   << " color=\""<<n.color<<"\","
	   << " fillcolor=\""<<n.fillcolor<<"\" "
	   << " ];"<<std::endl;

    // define the params of op
	BOOST_FOREACH(Op::param_t& p, o->m_params){
        std::string pstr = opstr + ":p" + boost::lexical_cast<std::string>(p->param_number);
        if(m_seen.find(p.get())==m_seen.end()){
            m_seen[p.get()] = pstr;
            std::string wd, nd = p->need_derivative ? "green" : "white";
            if(!is_sink)
                wd = boost::lexical_cast<std::string>(o->need_result() ? 4.0 : 0.5);
            else
                wd = boost::lexical_cast<std::string>(sink_prev->need_result() ? 4.0 : 0.5);

            std::string shape;
            if(m_verbose)
            if(p->shape.size()){
                shape = " (";
                for(unsigned int i=0;i<p->shape.size();i++){
                    shape += boost::lexical_cast<std::string>(p->shape[i]) + ((i==p->shape.size()-1) ? " " : ", ");
                }
                shape += ")";
            }

            //m_node_defs << pstr
                //<< " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\", width=0.01, label=\"" << p->param_number << shape<< "\", shape=circle ] ;"<<std::endl;
            //m_node_defs << pstr
                //<< " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\", width=0.01, label=\"" << p->param_number << shape<< "\", shape=circle ] ;"<<std::endl;
            // connect to op
            //m_edge_defs << "edge [style=solid,dir=none,penwidth="<<wd<<",weight=100] "<<pstr<<" -> "<<opstr<<";"<<std::endl;

            std::string vstr = define_data_ptr(p->value);
            if(vstr.size())
                m_edge_defs << "edge [style=dotted,dir=none,penwidth=.5,weight=10]" << pstr << " -> "<< vstr << "; "<<std::endl;
        }
    }
    // define the results of op
    BOOST_FOREACH(Op::result_t& r, o->m_results){
        std::string rstr = opstr + ":r" + boost::lexical_cast<std::string>(r->result_number);
        if(m_seen.find(r.get())==m_seen.end()){
            m_seen[r.get()] = rstr;
            std::string nd = r->need_result ? "gold1" : "white";
            std::string wd = boost::lexical_cast<std::string>(r->need_result ? 4.0 : 0.5);
            //m_node_defs << opstr << ":r"<<r->result_number
                //<< " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\",width=0.01, label=\"" << r->result_number << "\", shape=circle ] ;"<<std::endl;
            // connect to op
            //m_node_defs << "edge [style=solid,penwidth="<<wd<<",dir=none,weight=100] "<< opstr << " -> " << rstr <<";"<<std::endl;

            std::string vstr = define_data_ptr(r->delta);
            if(vstr.size())
                m_edge_defs << "edge [style=dotted,penwidth=.5,dir=none,weight=10]" << rstr << " -> "<< vstr << "; "<<std::endl;

            if(current_op()==o){
                // create node with debug info in label
                m_node_defs << opstr << "_dbg_result"
                    << " [ fontsize=26, label=\"";
                if(r->result_uses.size()>0){
                    if(r->result_uses[0].lock()->value.ptr()){
                        m_node_defs<<"Result:\\n";
                        bool has_nan = cuv::has_nan(r->result_uses[0].lock()->value.cdata());
                        bool has_inf = cuv::has_inf(r->result_uses[0].lock()->value.cdata());
                        m_node_defs  << "min: " << cuv::minimum(r->result_uses[0].lock()->value.cdata()) <<"\\n"
                            << "max: " << cuv::maximum(r->result_uses[0].lock()->value.cdata()) <<"\\n"
                            << "avg: " << cuv::mean(r->result_uses[0].lock()->value.cdata()) <<"\\n"
                            << "var: " << cuv::var(r->result_uses[0].lock()->value.cdata()) <<"\\n"
                            << "nan: " << has_nan <<"\\n"
                            << "inf: " << has_inf<<"\\n";
                        if(has_nan || has_inf)
                            m_break_after_done = true;
                    }

                    if(r->delta.ptr())
                    {
                        m_node_defs<<"Delta:\\n";
                        bool has_nan = cuv::has_nan(r->delta.cdata());
                        bool has_inf = cuv::has_inf(r->delta.cdata());
                        m_node_defs  << "min: " << cuv::minimum(r->delta.cdata()) <<"\\n"
                            << "max: " << cuv::maximum(r->delta.cdata()) <<"\\n"
                            << "avg: " << cuv::mean(r->delta.cdata()) <<"\\n"
                            << "var: " << cuv::var(r->delta.cdata()) <<"\\n"
                            << "nan: " << has_nan <<"\\n"
                            << "inf: " << has_inf<<"\\n";
                        if(has_nan || has_inf)
                            m_break_after_done = true;
                    }
                }
                m_node_defs << "\" ];"<<std::endl;
                m_edge_defs << "edge [style=solid,dir=none,weight=100] "<<opstr << " -> " << opstr<<"_dbg_result ;"<<std::endl;
            }
        }

    }

	BOOST_FOREACH(Op::param_t& p, o->m_params){
		BOOST_FOREACH(Op::result_t& r, p->param_uses){
            std::string wd;
            if(!is_sink)
                wd = boost::lexical_cast<std::string>(o->need_result() ? 4.0 : 0.5);
            else
                wd = boost::lexical_cast<std::string>(sink_prev->need_result() ? 4.0 : 0.5);
			m_edge_defs << "edge [ "
               << "dir=forward,"
               << "style=solid,"
               << "penwidth="<<wd<<","
               << "weight=1"
				   //<< " headlabel=\""<<boost::lexical_cast<std::string>(cnt) << "\""
			   <<" ] ";

            // draw a line from the result of an op to the param this result is used in
			m_edge_defs << "n" << boost::lexical_cast<std::string>( r->get_op().get() )
               << ":r" << r->result_number;

            detail::graphviz_node pn, rn;
            o->_graphviz_node_desc(pn);
            r->get_op()->_graphviz_node_desc(rn);
			m_edge_defs 
                << " -> " << opstr << ":p" << p->param_number << " ; "
                << " // pgroup=" << o->get_group() 
                << " rgroup=" << r->get_op()->get_group() 
                << " filter="<<m_group_filter
                << " plabel="<< pn.label
                << " rlabel="<< rn.label
                <<std::endl;
            
			//m_edge_defs << "n" << boost::lexical_cast<std::string>( (size_t)(r->get_op().get()) );
			//m_edge_defs << " -> ";
			//m_edge_defs << "n" << boost::lexical_cast<std::string>( (size_t) o );
			//m_edge_defs << " ; "<<std::endl;
		}
	}
}
void define_graphviz_node_visitor::postorder(Op* o){
}



void determine_exec_order::init(Op* o, int o_res,
        const std::vector<std::pair<Op*, int> >& results,
        const std::vector<Op*>& parameters){
    reset_needed_flags rnf;
    o->visit(rnf, true); 

    std::vector<std::pair<Op*, int> > empty;
    determine_fprop_list(o, o_res, empty);
    container_type     plainfprop = fprop_nodelist;
    determine_bprop_list(o, parameters, plainfprop);
    if(results.size() > 0)
        determine_fprop_list(o, o_res, results);
}

void determine_exec_order::determine_bprop_list(Op* o, const std::vector<Op*>& parameters, const std::vector<Op*>& plainfprop){
    bprop_nodelist.clear();
    std::list<Op*> queue;
    std::map<Op*, bool> marked;

    // we need to work forward from all these
    BOOST_FOREACH(Op* p, parameters){
        queue.push_back(p);
    }

    // difficulty here, as oppposed to fprop: 
    // - not all results are calculated in fprop
    // - not all results need bprop (actually, just one should...)
    while(!queue.empty()){
        Op* top = queue.back();
        queue.pop_back();

        // stop processing if we've seen this item before
        if(marked.find(top) != marked.end())
            continue;

        // stop processing if we diverge from the path leading to the loss we derived for.
        if(std::find(plainfprop.begin(), plainfprop.end(), top) 
                == plainfprop.end())
            continue;

        // check all results of the current one, whether they have been
        // calculated already 
        bool all_res_ready = true;
        BOOST_FOREACH(Op::result_t& p, top->m_results){
            for(unsigned int i=0;i<p->result_uses.size();i++) {
                auto r = p->use(i);
                r->need_derivative = true;
                Op* top_res = r->get_op();

                // only stuff that we need for plainfprop needs to be calculated already!
                if(std::find(plainfprop.begin(), plainfprop.end(), top_res) 
                        == plainfprop.end())
                    continue;
                if(marked.find(top_res) != marked.end()) {
                    continue;
                }
                // we need to calculate this before top!
                queue.push_back(top_res); 
                all_res_ready = false;
                break;
            }
            if(!all_res_ready)
                break;
        }
        if(all_res_ready){
            // calculate top at this position
            bprop_nodelist.push_back(top);
            top->need_derivative(true);
            marked[top] = true;
        }else{
            // need to revisit top at some later point in time
            queue.push_front(top);
        }
    }
}

/**
 * search all results of an op for a specific op instance.
 */
bool determine_exec_order::find_in_results(
        Op* query, 
        const std::vector<std::pair<Op*, int> >& other_queries,
        Op* search_start){
    std::list<Op*> queue;
    std::map<Op*, bool> marked;
    queue.push_back(search_start);

    while(!queue.empty()){
        Op* top = queue.back();
        if(marked[top]) {
            queue.pop_back();
            continue;
        }
        marked[top] = true;
        queue.pop_back();
        if(top == query)
            return true;
        for(std::vector<std::pair<Op*, int> >::const_iterator it = other_queries.begin();
                it != other_queries.end(); ++it)
            if(it->first == top)
                return true;
        BOOST_FOREACH(Op::result_t& r, top->m_results){
            for(unsigned int i=0;i<r->result_uses.size();i++) {
                queue.push_back(r->use(i)->get_op());
            }
        }
    }
    return false;
}

void determine_exec_order::determine_fprop_list(Op* o, int o_res, const std::vector<std::pair<Op*, int> >& results){
    // put all outputs in a queue
    // for every element in the queue,
    // - if all its predecessors have already been calculated,
    //   - put the element in the fprop_nodelist
    //   - mark it as done
    // - else
    //   - put the element in the queue again for visiting later
    fprop_nodelist.clear();
    std::list<Op*> queue;
    std::map<Op*, bool> marked;

    // we need to work backwards from all these
    typedef std::pair<Op*, int> pair_t;
    BOOST_FOREACH(const pair_t& pt, results){
        queue.push_back(pt.first);
    }
    queue.push_back(o); // make sure this is processed FIRST!
    o->result(o_res)->need_result = true;

    while(!queue.empty()){
        Op* top = queue.back();
        queue.pop_back();

        // stop processing if we've seen this item before
        if(marked.find(top) != marked.end())
            continue;

        // check all predecessors of the current one, whether they have been
        // calculated already
        bool all_preds_ready = true;
        BOOST_FOREACH(Op::param_t& p, top->m_params){
            for(unsigned int i=0;i<p->param_uses.size();i++) {
                auto r = p->use(i);
                r->need_result = true;
                Op* top_pred = r->get_op().get();
                if(marked.find(top_pred) != marked.end()) {
                    continue;
                }
                // we need to calculate this before top!
                queue.push_back(top_pred); 
                all_preds_ready = false;
                break;
            }
            if(!all_preds_ready)
                break;
        }
        if(all_preds_ready){
            // calculate top at this position
            fprop_nodelist.push_back(top);
            top->need_result(true);
            marked[top] = true;
        }else{
            // need to revisit top at some later point in time
            queue.push_front(top);
        }
    }
}


/**
 * @return true if the op is a ParameterInput and cannot be derived.
 */
static bool invalid_parameter(Op* op){
    bool is_deltasink = dynamic_cast<DeltaSink*>(op);
    if(is_deltasink)
        return false;

    ParameterInput* p = dynamic_cast<ParameterInput*>(op);
    if(p)
        return !p->derivable();

    // don't know what to do with this op!
    cuvAssert(false);
    return true;
}

swiper::~swiper(){
    if(m_cleanup_temp_vars){
	cleanup_temp_vars_visitor ctvv;
	m_op->visit(ctvv,true);
    }

    reset_needed_flags rnf;
    m_op->visit(rnf); 
}
void swiper::init()
{
    Op& op = *m_op;

    // clean paramlist from non-derivable parameters
    assert(count_if(m_paramlist.begin(), m_paramlist.end(), std::logical_not<bool>()) == 0);

    m_paramlist.erase(
            std::remove_if(m_paramlist.begin(), m_paramlist.end(), 
                invalid_parameter),
            m_paramlist.end());

    m_topo.init(&op, m_result, m_other_funcs, m_paramlist);
    check_param_existence();

    this->set_calculate_result();             // determine need_result

    cleanup_temp_vars_visitor ctvv;
    op.visit(ctvv); 

    determine_shapes(op);

    //if(m_verbosity>0)
    //    dump("swiper-initial.dot", m_verbosity>1);
}

void swiper::set_calculate_result(){
    BOOST_FOREACH(Op* op, m_topo.fprop_nodelist){
        BOOST_FOREACH(Op::result_t& r, op->m_results){
            r->determine_single_results();
        }
    }
    BOOST_FOREACH(Op* op, m_topo.bprop_nodelist){
        BOOST_FOREACH(Op::param_t& p, op->m_params){
            p->determine_single_results();
        }
    }
}

void swiper::request_other_result(Op& op, int result, bool call_init){
    m_other_funcs.push_back(std::make_pair(&op, result));
    if(call_init)
        init();
}

#define SWIPER_DEBUG 0
void swiper::fprop(){
	BOOST_FOREACH(Op* o, m_topo.fprop_nodelist){
		BOOST_FOREACH(Op::result_t& r, o->m_results){
			BOOST_FOREACH(Op::weak_param_t p, r->result_uses){
				p.lock()->value_set = false;
			}
		}
	}
    unsigned int cnt=0;
	BOOST_FOREACH(Op* o, m_topo.fprop_nodelist){
        if(o->need_result()){
            if(false){
                detail::graphviz_node n;
                o->_graphviz_node_desc(n);
                std::cout << cnt << ". fprop: "<<n.label<<";"<<std::endl;
            }
            o->fprop();
#if SWIPER_DEBUG
            debug(cnt,o,true,false,"fprop");
#endif
            cnt++;
        }
	}
}
void swiper::bprop(bool set_last_delta_to_one){
    if(set_last_delta_to_one){
        BOOST_FOREACH(Op::result_t& r, m_op->m_results){
            if(!r->delta)
                r->delta.reset(new Op::value_type(r->shape, Op::value_ptr::s_allocator));
            *r->delta = 1.f;
        }
    }

	BOOST_FOREACH(Op* o, m_topo.bprop_nodelist){
		BOOST_FOREACH(Op::param_t& p, o->m_params){
			BOOST_FOREACH(Op::result_t& r, p->param_uses){
				r->delta_set = false;
			}
		}
	}
#if SWIPER_DEBUG
    unsigned int cnt=0;
#endif
	BOOST_FOREACH(Op* o, m_topo.bprop_nodelist){
		if(o->need_derivative()){
#if SWIPER_DEBUG
            debug(cnt++,o,true,false,"bprop");
#endif
			o->bprop();
        }
	}
#if SWIPER_DEBUG
        std::cout << "Exiting after first bprop (swiper debugging)" << std::endl;
        exit(0);
#endif
}
void 
swiper::debug(unsigned int cnt, Op* o, bool results, bool params, const char* ident){
    if(results)
    {
        std::string s = boost::str(boost::format("dbg/%s-%03d")%ident%cnt);
        boost::filesystem::create_directories(s);
        std::ofstream os ((s+"/func.dot").c_str());
    
        write_graphviz(*m_topo.fprop_nodelist.back(),os,m_verbosity > 1,m_topo.fprop_nodelist,m_topo.bprop_nodelist,o);
    }
}


void swiper::check_param_existence()const{
    BOOST_FOREACH(Op* par, m_paramlist){
        if (std::find(m_topo.fprop_nodelist.begin(), m_topo.fprop_nodelist.end(), par) == m_topo.fprop_nodelist.end()){
            Input* inp = dynamic_cast<Input*>(par);
            if(inp)
                throw std::runtime_error("Parameter `" + inp->name() +  "' not found in the function fprop list");
            else
                throw std::runtime_error("Parameter not found in the function fprop list");
        }
        if (std::find(m_topo.bprop_nodelist.begin(), m_topo.bprop_nodelist.end(), par) == m_topo.bprop_nodelist.end()){
            Input* inp = dynamic_cast<Input*>(par);
            if(inp)
                throw std::runtime_error("Parameter `" + inp->name() +  "' not found in the function bprop list");
            else
                throw std::runtime_error("Parameter not found in the function bprop list");
        }
    }
} 

struct groups_collector
: public op_visitor_once_adaptor{
    std::vector<std::string> m_groups;

    inline void preorder(Op* o){
        if( !o->get_group().size() ||
                find(m_groups.begin(), m_groups.end(), o->get_group())
                != m_groups.end())
            return;
        m_groups.push_back(o->get_group());
    }
};

namespace cuvnet
{
void write_graphviz(Op& op, std::ostream& os, bool verbose){
	os << "digraph { "<<std::endl;
	os << "rankdir=TB; concentrate=true; remincross=true; splines=ortho; ranksep=1;"<<std::endl;
#if 1
    groups_collector gc;
    op.visit(gc);
    BOOST_FOREACH(std::string& g, gc.m_groups){
        os << "subgraph "<< g <<" { "<<std::endl;
        define_graphviz_node_visitor dgnv(verbose, NULL, NULL, g);
        op.visit(dgnv,true);
        os << "} "<<std::endl;
    }
    define_graphviz_node_visitor dgnv(verbose, NULL, NULL, "__empty");
	op.visit(dgnv,true);
#else
	define_graphviz_node_visitor dgnv(os);
	op.visit(dgnv,true);
#endif
	os << "}"<<std::endl;
    if(dgnv.m_break_after_done)
        exit(0);
}
void write_graphviz(Op& op, std::ostream& os, bool verbose, std::vector<Op*>& fl, std::vector<Op*>& bl, Op* current){
	os << "digraph { "<<std::endl;
	os << "rankdir=TB; concentrate=true; remincross=true; splines=ortho; ranksep=1;"<<std::endl;
#if 1
    groups_collector gc;
    op.visit(gc, true);
    std::ostringstream node_defs;
    std::ostringstream edge_defs;
    BOOST_FOREACH(std::string& g, gc.m_groups){
        node_defs << "subgraph cluster_"<< g <<" { style=filled; color=lightgrey; label=\"" << g <<"\";"<<std::endl;
        define_graphviz_node_visitor dgnv(verbose, &fl, &bl, g);
        dgnv.current_op(current);
        op.visit(dgnv,true);
        node_defs << dgnv.m_node_defs.str();
        node_defs << "} "<<std::endl;

        //edge_defs << "{ rank=same; " << std::endl << dgnv.m_edge_defs.str() << std::endl <<"}";
        edge_defs << dgnv.m_edge_defs.str() << std::endl;
    }
    {
        define_graphviz_node_visitor dgnv(verbose, &fl, &bl, "__empty");
        dgnv.current_op(current);
        op.visit(dgnv,true);
        node_defs << dgnv.m_node_defs.str();
        edge_defs << dgnv.m_edge_defs.str();
    }
#else
	define_graphviz_node_visitor dgnv(os, &fl, &bl);
    dgnv.current_op(current);
	op.visit(dgnv,true);
#endif

    os << node_defs.str() << edge_defs.str();

	os << "}"<<std::endl;
}

void valid_shape_info::determine_shapes(){
    LocalPooling* poolp;
    Convolve* convp;
    BedOfNails* bon;

    o2i_scale = 1.f;
    i_margin_l = 0.f; i_margin_r = 0.f;
    {
        // `it' points to the `input' object
        container_type::reverse_iterator it = plist.rbegin();

        while(it != plist.rend()){
            if((bon = dynamic_cast<BedOfNails*>(*it))){
                std::vector<unsigned int> inshape  = bon->param(0)->shape;
                std::vector<unsigned int> outshape = bon->result(0)->shape;
                i_margin_l += o2i_scale * bon->startx();
                i_margin_r += o2i_scale * ((inshape[1] - bon->startx()) 
                        - outshape[1] * bon->stridex());
#define VALID_SHAPE_INFO_DEBUG 0
#if VALID_SHAPE_INFO_DEBUG
                cuvAssert(inshape[1] == 
                        o2i_scale * bon->startx() 
                        + outshape[1] * bon->stridex()
                        + o2i_scale * ((inshape[1] - bon->startx()) 
                        - outshape[1] * bon->stridex()));
                cuvAssert(inshape[2] == 
                        o2i_scale * bon->startx() 
                        + outshape[2] * bon->stridex()
                        + o2i_scale * ((inshape[1] - bon->startx()) 
                        - outshape[2] * bon->stridex()));
#endif
                o2i_scale *= bon->stridex();
            }
            else if((poolp = dynamic_cast<LocalPooling*>(*it))){
                std::vector<unsigned int> inshape  = poolp->param(0)->shape;
                std::vector<unsigned int> outshape = poolp->result(0)->shape;
                i_margin_l += 0.f;
                i_margin_r += (inshape[1] - outshape[1] * poolp->stridex());
#if VALID_SHAPE_INFO_DEBUG
                cuvAssert(inshape[1] == 
                        0.f
                        + outshape[1] * poolp->stridex()
                        + (inshape[1]
                        - outshape[1] * poolp->stridex()));
                cuvAssert(inshape[2] == 
                        0.f
                        + outshape[2] * poolp->stridex()
                        + (inshape[1]
                        - outshape[1] * poolp->stridex()));
#endif
                o2i_scale *= poolp->stridex();
            }
            else if((convp = dynamic_cast<Convolve*>(*it))){
                cuvnet::determine_shapes(*convp);
                std::vector<unsigned int> inshape  = convp->param(0)->shape;
                std::vector<unsigned int> outshape = convp->result(0)->shape;

                int fs = std::sqrt(convp->param(1)->shape[1]);

                int lmarg = fs/2 + convp->padding_start();

                // left margin: fs/2 is removed from valid convolution           
                // but this can be compensated by (negative!) padding            
                i_margin_l += o2i_scale * lmarg;

                // this is the pixel in the input over which the center of the last filter
                int re = lmarg + outshape[1] * convp->stride();

                // the margin on the right hand side might even be negative, if the last
                // filter extends outside the image.
                i_margin_r += o2i_scale * ((int)inshape[1] - re);

#if VALID_SHAPE_INFO_DEBUG
                cuvAssert(inshape[1] == 
                        lmarg
                        + outshape[1] * convp->stride()
                        + ((int)inshape[1] - re));
                cuvAssert(inshape[2] == 
                        lmarg
                        + outshape[2] * convp->stride()
                        + ((int)inshape[1] - re));
#endif
                // adjust scale so that margins in the next iteration can be calculated correctly
                o2i_scale *= convp->stride();

                // to place an output in the original image, proceed as follows:
                // 1. in the original image, extract ROI defined by margins
                // 2. upscale output so that its size matches that ROI.

            }

            it = it+1;
        }
    }
    /*
     * // `it' points to the `output' object
     *container_type::iterator it = plist.begin();
     *determine_shapes_visitor dsv;
     *(*it)->visit(dsv);
     *std::vector<unsigned int> outshape = (*it)->result(0)->shape;
     *cuvAssert(outshape.size() == 4);
     */


/*
 *    while(*it != plist.back()){
 *        if((bon = dynamic_cast<BedOfNails*>(*it))){
 *            std::vector<unsigned int> inshape  = bon->param(0)->shape;
 *            std::vector<unsigned int> outshape = bon->result(0)->shape;
 *            crop_h  *= inshape[1] / outshape[1];
 *            crop_w  *= inshape[2] / outshape[2];
 *            scale_h *= inshape[1] / outshape[1];
 *            scale_w *= inshape[2] / outshape[2];
 *        }
 *        else if((poolp = dynamic_cast<LocalPooling*>(*it))){
 *            std::vector<unsigned int> inshape  = poolp->param(0)->shape;
 *            std::vector<unsigned int> outshape = poolp->result(0)->shape;
 *            crop_h  *= inshape[1] / outshape[1];
 *            crop_w  *= inshape[2] / outshape[2];
 *            scale_h *= inshape[1] / outshape[1];
 *            scale_w *= inshape[2] / outshape[2];
 *        }
 *        else if((convp = dynamic_cast<Convolve*>(*it))){
 *            std::vector<unsigned int> inshape  = convp->param(0)->shape;
 *            std::vector<unsigned int> outshape = convp->result(0)->shape;
 *            // `valid' convolution amounts to /cropping/
 *            int oh = outshape[1] * convp->stride();
 *            int ow = outshape[2] * convp->stride();
 *            scale_h *= convp->stride();
 *            scale_w *= convp->stride();
 *            if(convp->is_padded()){
 *                // TODO assumes symmetric padding
 *                crop_h += inshape[1] - (oh+1 - convp->padding_size());
 *                crop_w += inshape[2] - (ow+1 - convp->padding_size());
 *            }else{
 *                crop_h += inshape[1] - oh+1;
 *                crop_w += inshape[2] - ow+1;
 *            }
 *        }
 *
 *        it = it+1;
 *    }
 */
}

std::pair<float, float> 
valid_shape_info::o2i(float y, float x)const{
    y *= o2i_scale;
    y += i_margin_l;

    x *= o2i_scale;
    x += i_margin_l;
    return std::make_pair(y, x);
}

std::pair<float, float> 
valid_shape_info::i2o(float y, float x)const{
    y -= i_margin_l;
    y /= o2i_scale;

    x -= i_margin_l;
    x /= o2i_scale;
    return std::make_pair(y, x);
}

}
