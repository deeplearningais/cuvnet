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
    if(!o->need_derivative())
        n.fillcolor = "white";
    else
        n.fillcolor = "gray70";
	n.style = "filled";
    //n.label += boost::lexical_cast<std::string>(o);
    if(dynamic_cast<Input*>(o)){
        // TODO: should check need_fprop, not need_bprop
        //if(o->need_derivative())
            //n.fillcolor = "goldenrod3";
        //else
            n.fillcolor = "lemonchiffon1";

        Op::value_ptr p = ((Input*)o)->data_ptr();
        if(!p);
        else{
            std::ostringstream ss;
            ss<<" (";
            for(unsigned int i=0;i<p.cdata().ndim();i++)
                ss<<p.cdata().shape(i)<<",";
            ss<<")";
            n.label += ss.str();
        }
    }else if(dynamic_cast<Sink*>(o)){
        //if(o->need_derivative())
            //n.fillcolor = "cadetblue";
        //else
            n.fillcolor = "lightcyan";
    }
	o->_graphviz_node_desc(n);

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
            std::string nd = p->need_derivative ? "green" : "white";
            os << pstr
                << " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\", width=0.01, label=\"" << p->param_number << "\", shape=circle ] ;"<<std::endl;
            // connect to op
            os << "edge [style=solid,dir=none,weight=100] "<<pstr<<" -> "<<opstr<<";"<<std::endl;

            std::string vstr = define_data_ptr(p->value);
            if(vstr.size())
                os << "edge [style=dotted,dir=none,weight=10]" << pstr << " -> "<< vstr << "; "<<std::endl;
        }
    }
    // define the results of op
    BOOST_FOREACH(Op::result_t& r, o->m_results){
        std::string rstr = opstr + "r" + boost::lexical_cast<std::string>(r->result_number);
        if(m_seen.find(r.get())==m_seen.end()){
            m_seen[r.get()] = rstr;
            std::string nd = r->need_result ? "gold1" : "white";
            os << opstr << "r"<<r->result_number
                << " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\",width=0.01, label=\"" << r->result_number << "\", shape=circle ] ;"<<std::endl;
            // connect to op
            os << "edge [style=solid,dir=none,weight=100] "<< opstr << " -> " << rstr <<";"<<std::endl;

            std::string vstr = define_data_ptr(r->delta);
            if(vstr.size())
                os << "edge [style=dotted,dir=none,weight=10]" << rstr << " -> "<< vstr << "; "<<std::endl;

            if(current_op()==o){
                // create node with debug info in label
                os << opstr << "_dbg"
                    << " [ fontsize=16, label=\"";
                if(r->result_uses.size()>0){
                    bool has_nan = cuv::has_nan(r->result_uses[0].lock()->value.cdata());
                    bool has_inf = cuv::has_inf(r->result_uses[0].lock()->value.cdata());
                    os  << "min: " << cuv::minimum(r->result_uses[0].lock()->value.cdata()) <<"\\n"
                        << "max: " << cuv::maximum(r->result_uses[0].lock()->value.cdata()) <<"\\n"
                        << "avg: " << cuv::mean(r->result_uses[0].lock()->value.cdata()) <<"\\n"
                        << "nan: " << has_nan <<"\\n"
                        << "inf: " << has_inf<<"\\n";
                    if(has_nan || has_inf)
                        m_break_after_done = true;
                }
                os << "\" ];"<<std::endl;
                os << "edge [style=solid,dir=none,weight=100] "<<opstr << " -> " << opstr<<"_dbg ;"<<std::endl;
            }
        }

    }
}
void define_graphviz_node_visitor::postorder(Op* o){
	unsigned int cnt = 0;
	BOOST_FOREACH(Op::param_t& p, o->m_params){
		BOOST_FOREACH(Op::result_t& r, p->param_uses){
			os << "edge [ "
               << "dir=forward,"
               << "style=solid,"
               << "penwidth="+boost::lexical_cast<std::string>(r->need_result ? 2.0 : 0.5)+","
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
		o->fprop();
#if SWIPER_DEBUG
        debug(cnt++,o,true,false);
#endif
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
	BOOST_REVERSE_FOREACH(Op* o, m_topo.plist){
		if(o->need_derivative())
			o->bprop();
	}
}
void 
swiper::debug(unsigned int cnt, Op* o, bool results, bool params){
    if(results)
    {
        std::string s = boost::str(boost::format("dbg/fprop-%03d")%cnt);
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
