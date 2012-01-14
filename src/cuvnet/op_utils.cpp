#include <fstream>
#include <boost/lexical_cast.hpp>
#include "op_utils.hpp"


using namespace cuvnet;

std::string define_graphviz_node_visitor::define_data_ptr(const Op::value_ptr& p){
    if(!p)
        return "";
    if(m_seen.find(&p.cdata())!=m_seen.end())
        return m_seen[&p.cdata()];
    m_seen[&p.cdata()] = "v"+boost::lexical_cast<std::string>((size_t)&p.cdata());
    os << m_seen[&p.cdata()] << " [ label=\"(";
    std::copy(p.cdata().shape().begin(),p.cdata().shape().end(),std::ostream_iterator<unsigned int>(os,","));
    os  << ")\" ] ; "<<std::endl;
    return m_seen[&p.cdata()];
}
void define_graphviz_node_visitor::preorder(Op* o){

	// fill in defaults
	detail::graphviz_node n;
	n.shape = "box";
	n.color = "black";
	n.fillcolor = "gray92";
	n.style = "filled";
	o->_graphviz_node_desc(n);

	if(m_mark_order.size()){
		std::vector<Op*>::iterator it = std::find(m_mark_order.begin(),m_mark_order.end(),o);
		if(it!=m_mark_order.end())
			n.label += " (" + boost::lexical_cast<std::string>(std::distance(m_mark_order.begin(),it))+")";
	}

    // define the op-node itself
    std::string opstr = "n" + boost::lexical_cast<std::string>( (size_t)(o) );
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
            std::string nd = p->need_derivative ? "green" : "white";
            m_seen[p.get()] = pstr;
            os << pstr
                << " [ style=filled, fillcolor="<<nd<<", fontsize=6, margin=\"0,0\", width=0.01, label=\"" << p->param_number << "\", shape=circle ] ;"<<std::endl;
            // connect to op
            os << "edge [style=solid,dir=none,weight=100] "<<pstr<<" -> "<<opstr<<";"<<std::endl;

            std::string vstr = define_data_ptr(p->value);
            if(vstr.size())
                os << "edge [style=dotted,dir=none,weight=0.2]" << pstr << " -> "<< vstr << "; "<<std::endl;
        }

    }
    // define the results of op
    BOOST_FOREACH(Op::result_t& r, o->m_results){
        std::string rstr = opstr + "r" + boost::lexical_cast<std::string>(r->result_number);
        if(m_seen.find(r.get())==m_seen.end()){
            m_seen[r.get()] = rstr;
            os << opstr << "r"<<r->result_number
                << " [ style=filled, fillcolor=gold1, fontsize=6, margin=\"0,0\",width=0.01, label=\"" << r->result_number << "\", shape=circle ] ;"<<std::endl;
            // connect to op
            os << "edge [style=solid,dir=none,weight=100] "<< opstr << " -> " << rstr <<";"<<std::endl;

            std::string vstr = define_data_ptr(r->delta);
            if(vstr.size())
                os << "edge [style=dotted,dir=none,weight=0.2]" << rstr << " -> "<< vstr << "; "<<std::endl;
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
               << "weight=0.1"
				   //<< " headlabel=\""<<boost::lexical_cast<std::string>(cnt) << nd<< "\""
			   <<" ]"<<std::endl;

			os << "n" << boost::lexical_cast<std::string>( (size_t)(r->get_op().get()) )
               << "r" << r->result_number;
			os << " -> ";
			os << "n" << boost::lexical_cast<std::string>( (size_t) o )
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

void swiper::fprop(){
	BOOST_FOREACH(Op* o, m_topo.plist){
		BOOST_FOREACH(Op::result_t& r, o->m_results){
			BOOST_FOREACH(Op::weak_param_t p, r->result_uses){
				p.lock()->value_set = false;
			}
		}
	}
	BOOST_FOREACH(Op* o, m_topo.plist){
		o->fprop();
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
void cuvnet::write_graphviz(Op& op, std::ostream& os){
	os << "digraph { "<<std::endl;
	define_graphviz_node_visitor dgnv(os);
	op.visit(dgnv);
	os << "}"<<std::endl;
}
void cuvnet::write_graphviz(Op& op, std::ostream& os, std::vector<Op*>& l){
	os << "digraph { "<<std::endl;
	define_graphviz_node_visitor dgnv(os,&l);
	op.visit(dgnv);
	os << "}"<<std::endl;
}
