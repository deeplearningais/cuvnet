#include "op.hpp"

using namespace cuvnet;


Op::Op(){}

Op::Op(unsigned int n_params, unsigned int n_results){
    set_n_params(n_params);
    set_n_results(n_results);
}

Op::~Op(){
	detach_from_params();
	detach_from_results();
}

Op& Op::detach_from_params()
{
    BOOST_FOREACH(Op::param_t& p, m_params){
	p->clear();
    }
    return *this;
}
Op& Op::detach_from_results(){
	BOOST_FOREACH(Op::result_t& r, m_results){
		r->clear();
	}
	return *this;
}

Op::result_t&
Op::result(const unsigned int i){
    if(!m_results[i]->op)
	m_results[i]->op = shared_from_this();
    return m_results[i];
}
Op::param_t&       
Op::param(const unsigned int i){
    return m_params[i];
}
void 
Op::set_n_params(unsigned int n){ 
	m_params.resize(n); 
	for(unsigned int i=0;i<n;i++){
		m_params[i].reset(new detail::op_param<value_type>());
        m_params[i]->param_number = i;
	}
}
void 
Op::set_n_results(unsigned int n){ 
    m_results.resize(n); 
    for(unsigned int i=0;i<n;i++){
        m_results[i].reset(new detail::op_result<value_type>());
        m_results[i]->result_number = i;
    }
}
void 
Op::add_param(unsigned int idx, result_t& p){
    param(idx)->param_uses.push_back(p);
    p->result_uses.push_back(param(idx));
}
bool 
Op::set_calculate_derivative(const std::vector<Op*>&l){
	if(l.end() != std::find(l.begin(),l.end(), this)){
		assert(m_params.size()==0); // this should be a "scalar"
		return true;
	}
	bool need_calc_derivative = false;
	BOOST_FOREACH(param_t& p, m_params){
		bool derive_wrt_p = false;
		BOOST_FOREACH(Op::result_t& r, p->param_uses){
			derive_wrt_p |= r->get_op()->set_calculate_derivative(l);
		}
		p->need_derivative = derive_wrt_p;
		need_calc_derivative |= derive_wrt_p;
	}
	m_need_derivative = need_calc_derivative;
	return need_calc_derivative;
}

void Op::_determine_shapes(){
	assert(m_params.size()==1);
	BOOST_FOREACH(result_t& r, m_results){
		r->shape = m_params[0]->shape;
	}
}
