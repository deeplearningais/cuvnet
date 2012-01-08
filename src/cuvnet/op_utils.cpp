#include <boost/lexical_cast.hpp>
#include "op_utils.hpp"


using namespace cuvnet;

void define_graphviz_node_visitor::preorder(Op* o){
	os << "n" << boost::lexical_cast<std::string>( (size_t)(o) );
	os << " ["
	   << o->_graphviz_node_desc()
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
		o->bprop();
	}
}
void cuvnet::write_graphviz(Op& op, std::ostream& os){
	os << "digraph { "<<std::endl;
	define_graphviz_node_visitor dgnv(os);
	op.visit(dgnv);
	os << "}"<<std::endl;
}
