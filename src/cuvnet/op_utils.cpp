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
    os << m_seen[&p.cdata()] << " [ label=\"(";
    std::vector<unsigned int> shape = p.cdata().shape();
    std::copy(shape.begin(),shape.end(),std::ostream_iterator<unsigned int>(os,","));
    os  << ")\" ] ; "<<std::endl;
    return m_seen[&p.cdata()];
}
void define_graphviz_node_visitor::preorder(Op* o){

	// fill in defaults
	detail::graphviz_node n;
	n.shape = "box";
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
#ifndef NDEBUG
    n.label += " " + boost::lexical_cast<std::string>(o);
#endif
    if(o->get_label().size())
    {
        n.label = n.label + "\\n"+ o->get_label();
        n.penwidth = 4.f;
    }
    if(o->get_group().size()){
        n.group = o->get_group();
    }

	if(m_fmark_order.size()){
		std::vector<Op*>::iterator fit = std::find(m_fmark_order.begin(),m_fmark_order.end(),o);
		std::vector<Op*>::iterator bit = std::find(m_bmark_order.begin(),m_bmark_order.end(),o);
#ifndef NDEBUG
		if(fit!=m_fmark_order.end() && bit!=m_bmark_order.end())
			n.label += " <" + boost::lexical_cast<std::string>(std::distance(m_fmark_order.begin(),fit))
			+ ", " + boost::lexical_cast<std::string>(std::distance(m_bmark_order.begin(),bit))+">";
        else if(fit!=m_fmark_order.end())
			n.label += " <" + boost::lexical_cast<std::string>(std::distance(m_fmark_order.begin(),fit))+",>";
        else if(bit!=m_bmark_order.end())
			n.label += " <," + boost::lexical_cast<std::string>(std::distance(m_bmark_order.begin(),bit))+">";
#endif
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
    m_seen[o] = opstr;
	os << opstr
	   << " ["
	   << " URL=\""<<type<<boost::lexical_cast<std::string>(o)<<"\","
	   << " label=\""<<n.label<<"\","
	   << groupstr
	   << " shape=\""<<n.shape<<"\","
	   << " style=\""<<n.style<<"\","
	   << " penwidth=\""<<n.penwidth<<"\","
	   << " color=\""<<n.color<<"\","
	   << " fillcolor=\""<<n.fillcolor<<"\" "
	   << " ];"<<std::endl;

    // define the params of op
	BOOST_FOREACH(Op::param_t& p, o->m_params){
        std::string pstr = opstr + "p" + boost::lexical_cast<std::string>(p->param_number);
        if(m_seen.find(p.get())==m_seen.end()){
            m_seen[p.get()] = pstr;
            std::string wd, nd = p->need_derivative ? "green" : "white";
            if(!is_sink)
                wd = boost::lexical_cast<std::string>(o->need_result() ? 4.0 : 0.5);
            else
                wd = boost::lexical_cast<std::string>(sink_prev->need_result() ? 4.0 : 0.5);

            std::string shape;
#ifndef NDEBUG
            if(p->shape.size()){
                shape = " (";
                for(unsigned int i=0;i<p->shape.size();i++){
                    shape += boost::lexical_cast<std::string>(p->shape[i]) + ((i==p->shape.size()-1) ? " " : ", ");
                }
                shape += ")";
            }
#endif

            os << pstr
                << " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\", width=0.01, label=\"" << p->param_number << shape<< "\", shape=circle ] ;"<<std::endl;
            // connect to op
            os << "edge [style=solid,dir=none,penwidth="<<wd<<",weight=100] "<<pstr<<" -> "<<opstr<<";"<<std::endl;

            std::string vstr = define_data_ptr(p->value);
            if(vstr.size())
                os << "edge [style=dotted,dir=none,penwidth=.5,weight=10]" << pstr << " -> "<< vstr << "; "<<std::endl;
        }
    }
    // define the results of op
    BOOST_FOREACH(Op::result_t& r, o->m_results){
        std::string rstr = opstr + "r" + boost::lexical_cast<std::string>(r->result_number);
        if(m_seen.find(r.get())==m_seen.end()){
            m_seen[r.get()] = rstr;
            std::string nd = r->need_result ? "gold1" : "white";
            std::string wd = boost::lexical_cast<std::string>(r->need_result ? 4.0 : 0.5);
            os << opstr << "r"<<r->result_number
                << " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\",width=0.01, label=\"" << r->result_number << "\", shape=circle ] ;"<<std::endl;
            // connect to op
            os << "edge [style=solid,penwidth="<<wd<<",dir=none,weight=100] "<< opstr << " -> " << rstr <<";"<<std::endl;

            std::string vstr = define_data_ptr(r->delta);
            if(vstr.size())
                os << "edge [style=dotted,penwidth=.5,dir=none,weight=10]" << rstr << " -> "<< vstr << "; "<<std::endl;

            if(current_op()==o){
                // create node with debug info in label
                os << opstr << "_dbg_result"
                    << " [ fontsize=26, label=\"";
                if(r->result_uses.size()>0){
                    if(r->result_uses[0].lock()->value.ptr()){
                        os<<"Result:\\n";
                        bool has_nan = cuv::has_nan(r->result_uses[0].lock()->value.cdata());
                        bool has_inf = cuv::has_inf(r->result_uses[0].lock()->value.cdata());
                        os  << "min: " << cuv::minimum(r->result_uses[0].lock()->value.cdata()) <<"\\n"
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
                        os<<"Delta:\\n";
                        bool has_nan = cuv::has_nan(r->delta.cdata());
                        bool has_inf = cuv::has_inf(r->delta.cdata());
                        os  << "min: " << cuv::minimum(r->delta.cdata()) <<"\\n"
                            << "max: " << cuv::maximum(r->delta.cdata()) <<"\\n"
                            << "avg: " << cuv::mean(r->delta.cdata()) <<"\\n"
                            << "var: " << cuv::var(r->delta.cdata()) <<"\\n"
                            << "nan: " << has_nan <<"\\n"
                            << "inf: " << has_inf<<"\\n";
                        if(has_nan || has_inf)
                            m_break_after_done = true;
                    }
                }
                os << "\" ];"<<std::endl;
                os << "edge [style=solid,dir=none,weight=100] "<<opstr << " -> " << opstr<<"_dbg_result ;"<<std::endl;
            }
        }

    }
}
void define_graphviz_node_visitor::postorder(Op* o){
	unsigned int cnt = 0;
    bool is_sink  = dynamic_cast<Sink*>(o);
    Op* sink_prev = NULL;
    if(is_sink)
        sink_prev = o->param(0)->use(0)->get_op().get();

	BOOST_FOREACH(Op::param_t& p, o->m_params){
		BOOST_FOREACH(Op::result_t& r, p->param_uses){
            std::string wd;
            if(!is_sink)
                wd = boost::lexical_cast<std::string>(o->need_result() ? 4.0 : 0.5);
            else
                wd = boost::lexical_cast<std::string>(sink_prev->need_result() ? 4.0 : 0.5);
			os << "edge [ "
               << "dir=forward,"
               << "style=solid,"
               << "penwidth="<<wd<<","
               << "weight=1"
				   //<< " headlabel=\""<<boost::lexical_cast<std::string>(cnt) << "\""
			   <<" ]"<<std::endl;

            // draw a line from the result of an op to the param this result is used in
			os << "n" << boost::lexical_cast<std::string>( r->get_op().get() )
               << "r" << r->result_number;
			os << " -> ";
			os << "n" << boost::lexical_cast<std::string>(  o )
			   << "p" << p->param_number;
			os << " ; "<<std::endl;
            
			//os << "n" << boost::lexical_cast<std::string>( (size_t)(r->get_op().get()) );
			//os << " -> ";
			//os << "n" << boost::lexical_cast<std::string>( (size_t) o );
			//os << " ; "<<std::endl;
		}
		cnt++;
	}
}



void determine_exec_order::init(Op* o, int o_res,
        const std::vector<std::pair<Op*, int> >& results,
        const std::vector<Op*>& parameters){
    reset_needed_flags rnf;
    o->visit(rnf, true); 

    std::vector<std::pair<Op*, int> > empty;
    determine_fprop_list(o, o_res, empty);
    determine_bprop_list(o, parameters);
    if(results.size() > 0)
        determine_fprop_list(o, o_res, results);
}

void determine_exec_order::determine_bprop_list(Op* o, const std::vector<Op*>& parameters){
    bprop_nodelist.clear();
    std::list<Op*> queue;
    std::map<Op*, bool> marked;

    // we need to work forward from all these
    BOOST_FOREACH(Op* p, parameters){
        queue.push_back(p);
    }

    while(!queue.empty()){
        Op* top = queue.back();
        queue.pop_back();

        // stop processing if we've seen this item before
        if(marked.find(top) != marked.end())
            continue;

        marked[top] = true;
        top->need_derivative(true);
        bprop_nodelist.push_back(top); // collect in reverse order

        // mark all successors of top as needed
        BOOST_FOREACH(Op::result_t& r, top->m_results){

            // do not calculate derivative for parts of the op that do
            // not need to be calculated during fprop
            if(!r->need_result)
                continue;

            // everyone else who uses this result must be derived
            for(unsigned int i=0;i<r->result_uses.size();i++) {
                auto p = r->use(i);
                Op* top_suc = p->get_op();
                if(! top_suc->need_result() )
                    continue;
                p->need_derivative = true;

                // we shall only push this to the queue if everyone who uses it
                // has already been put into the queue
                // this is annoying, but removing edges (as in the wp version)
                // is not an option.
                bool can_push = true;
                BOOST_FOREACH(Op::param_t& p2, top_suc->m_params){
                    for(unsigned int j = 0; j < p2->param_uses.size(); j++){
                        Op* p2_op = p2->use(j)->get_op().get();
                        if(marked.find(p2_op) == marked.end()) {

                            // check whether p2_op depends on any of the parameters
                            // if it depends, these need to be dealt with first
                            param_collector_visitor pcv;
                            p2_op->visit(pcv);
                            if( pcv.plist.end() !=
                                    std::find_first_of(
                                        pcv.plist.begin(), pcv.plist.end(),
                                        parameters.begin(), parameters.end())) {
                                can_push = false;
                                break;
                            }
                        }
                    }
                    if(!can_push)
                        break;
                }
                if(can_push)
                    queue.push_front(top_suc);
            }
        }
    }
    std::reverse(bprop_nodelist.begin(), bprop_nodelist.end());
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
        if(marked[top])
            continue;
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

        marked[top] = true;
        top->need_result(true);
        fprop_nodelist.push_back(top); // collect in reverse order

        // mark all predecessors of top as needed
        BOOST_FOREACH(Op::param_t& p, top->m_params){
            for(unsigned int i=0;i<p->param_uses.size();i++) {
                auto r = p->use(i);
                r->need_result = true;

                // we shall only push this to the queue if everyone who uses it
                // has already been put into the queue
                // this is annoying, but removing edges (as in the wp version)
                // is not an option.
                Op* top_pred = r->get_op().get();
                bool can_push = true;
                BOOST_FOREACH(Op::result_t& r2, top_pred->m_results){
                    for(unsigned int j = 0; j < r2->result_uses.size(); j++){
                        Op* target = r2->use(j)->get_op();
                        if(marked.find(target) == marked.end()) {
                            // if the tree below target does not result in any
                            // of the 'results' we're interested in, we can
                            // ignore it.
                            if(!find_in_results(o, results, target))
                                continue;
                            can_push = false;
                            break;
                        }
                    }
                    if(!can_push)
                        break;
                }
                if(can_push)
                    queue.push_front(top_pred);
            }
        }
    }
    std::reverse(fprop_nodelist.begin(), fprop_nodelist.end());
}


void swiper::init()
{
    Op& op = *m_op;

    // clean paramlist from non-derivable parameters
    m_paramlist.erase(
            std::remove_if(m_paramlist.begin(), m_paramlist.end(), 
                std::not1( // not [  cast_to_input(op)->derivable()   ]
                    __gnu_cxx::compose1(
                        std::mem_fun( &ParameterInput::derivable ),
                        ptr_caster<Op,ParameterInput>()))),
            m_paramlist.end());

    m_topo.init(&op, m_result, m_other_funcs, m_paramlist);
    check_param_existence();

    this->set_calculate_result();             // determine need_result
    op.set_calculate_derivative(m_paramlist); // determine need_derivative

    cleanup_temp_vars_visitor ctvv;
    op.visit(ctvv); 

    determine_shapes(op);

    dump("swiper-initial.dot");
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
        BOOST_FOREACH(Op::result_t& r, m_topo.fprop_nodelist.back()->m_results){
            if(!r->delta)
                r->delta.reset(new Op::value_type(r->shape));
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
    
        write_graphviz(*m_topo.fprop_nodelist.back(),os,m_topo.fprop_nodelist,m_topo.bprop_nodelist,o);
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

void cuvnet::write_graphviz(Op& op, std::ostream& os){
	os << "digraph { "<<std::endl;
	define_graphviz_node_visitor dgnv(os);
	op.visit(dgnv,true);
	os << "}"<<std::endl;
    if(dgnv.m_break_after_done)
        exit(0);
}
void cuvnet::write_graphviz(Op& op, std::ostream& os, std::vector<Op*>& fl, std::vector<Op*>& bl, Op* current){
	os << "digraph { "<<std::endl;
	define_graphviz_node_visitor dgnv(os, &fl, &bl);
    dgnv.current_op(current);
	op.visit(dgnv,true);
	os << "}"<<std::endl;
}


