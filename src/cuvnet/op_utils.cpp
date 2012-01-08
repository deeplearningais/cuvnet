#include <boost/lexical_cast.hpp>
#include "op_utils.hpp"


using namespace cuvnet;

void define_graphviz_node_visitor::preorder(Op* o){
	os << "n" << boost::lexical_cast<std::string>( (size_t)(o) );

	// fill in defaults
	detail::graphviz_node n;
	n.shape = "box";
	n.color = "white";
	o->_graphviz_node_desc(n);

	if(m_mark_order.size()){
		std::vector<Op*>::iterator it = std::find(m_mark_order.begin(),m_mark_order.end(),o);
		if(it!=m_mark_order.end())
			n.label += " (" + boost::lexical_cast<std::string>(std::distance(m_mark_order.begin(),it))+")";
	}

	os << " ["
	   << " label=\""<<n.label<<"\" "
	   << " shape=\""<<n.shape<<"\" "
	   << " ];"<<std::endl;
}
void define_graphviz_node_visitor::postorder(Op* o){
	unsigned int cnt = 0;
	BOOST_FOREACH(Op::param_t& p, o->m_params){
		BOOST_FOREACH(Op::result_t& r, p->param_uses){
			os << "edge [ "
		           << " headlabel=\""<<boost::lexical_cast<std::string>(cnt) << "\""
			   <<" ]"<<std::endl;
			os << "n" << boost::lexical_cast<std::string>( (size_t)(r->get_op().get()) );
			os << " -> ";
			os << "n" << boost::lexical_cast<std::string>( (size_t) o );
			os << " ; "<<std::endl;
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
void swiper::bprop(){
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
