#include <fstream>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include "op_utils.hpp"
#include <cuvnet/ops/output.hpp>


using namespace cuvnet;

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
    bool is_input = dynamic_cast<Input*>(o);
    bool is_sink  = dynamic_cast<Sink*>(o);
    Op* sink_prev = NULL;
    if(is_input){
        if(!o->need_result())
            n.fillcolor = "lemonchiffon1";
        else if(!o->need_derivative())
            n.fillcolor = "goldenrod1";
        else
            n.fillcolor = "goldenrod3";

        Op::value_ptr p = ((Input*)o)->data_ptr();
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

	if(m_mark_order.size()){
		std::vector<Op*>::iterator it = std::find(m_mark_order.begin(),m_mark_order.end(),o);
		if(it!=m_mark_order.end())
			n.label += " <" + boost::lexical_cast<std::string>(std::distance(m_mark_order.begin(),it))+">";
	}
    if(current_op()==o){
        n.fillcolor = "firebrick1";
    }

    // define the op-node itself
    std::string opstr = "n" + boost::lexical_cast<std::string>( o );
    m_seen[o] = opstr;
	os << opstr
	   << " ["
	   << " tooltip=\"tooltip "<<n.label<<"\","
	   << " label=\""<<n.label<<"\","
	   << " shape=\""<<n.shape<<"\","
	   << " style=\""<<n.style<<"\","
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
            if(p->shape.size()){
                shape = " (";
                for(unsigned int i=0;i<p->shape.size();i++){
                    shape += boost::lexical_cast<std::string>(p->shape[i]) + ((i==p->shape.size()-1) ? " " : ", ");
                }
                shape += ")";
            }

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

#define SWIPER_DEBUG 0
void swiper::fprop(){
	BOOST_FOREACH(Op* o, m_topo.plist){
        if(dynamic_cast<Sink*>(o) != NULL) // do not reset sinks
            continue;
		BOOST_FOREACH(Op::result_t& r, o->m_results){
			BOOST_FOREACH(Op::weak_param_t p, r->result_uses){
				p.lock()->value_set = false;
			}
		}
	}
#if SWIPER_DEBUG
    unsigned int cnt=0;
#endif
	BOOST_FOREACH(Op* o, m_topo.plist){
        if(o->need_result()){
            o->fprop();
#if SWIPER_DEBUG
            debug(cnt++,o,true,false,"fprop");
#endif
        }
	}
}
void swiper::bprop(bool set_last_delta_to_one){
    if(set_last_delta_to_one){
        BOOST_FOREACH(Op::result_t& r, m_topo.plist.back()->m_results){
            if(!r->delta)
                r->delta.reset(new Op::value_type(r->shape));
            *r->delta = 1.f;
        }
    }

	BOOST_FOREACH(Op* o, m_topo.plist){
		BOOST_FOREACH(Op::param_t& p, o->m_params){
			BOOST_FOREACH(Op::result_t& r, p->param_uses){
				r->delta_set = false;
			}
		}
	}
#if SWIPER_DEBUG
    unsigned int cnt=0;
#endif
	BOOST_REVERSE_FOREACH(Op* o, m_topo.plist){
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
    
        write_graphviz(*m_topo.plist.back(),os,m_topo.plist,o);
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
void cuvnet::write_graphviz(Op& op, std::ostream& os, std::vector<Op*>& l, Op* current){
	os << "digraph { "<<std::endl;
	define_graphviz_node_visitor dgnv(os,&l);
    dgnv.current_op(current);
	op.visit(dgnv,true);
	os << "}"<<std::endl;
}
