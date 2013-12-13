#include "op.hpp"
#include <cuvnet/ops/output.hpp>

using namespace cuvnet;

namespace 
{
    std::map<std::string, int> g_group_idx;
    std::vector<std::string> g_groups;
}
op_group::op_group(const std::string& name, bool uniq){
    if(uniq)
        g_groups.push_back(name + "_" + boost::lexical_cast<std::string>(g_group_idx[name]++));
    else
        g_groups.push_back(name);
}
op_group::~op_group(){
    g_groups.pop_back();
}

Op::Op(){}

Op::Op(unsigned int n_params, unsigned int n_results)
    :m_need_derivative(false)
    ,m_need_result(false)
{
    if(g_groups.size())
        m_group = g_groups.back();
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
    if( ! param(idx)->op)
        param(idx)->op = this;
    param(idx)->param_uses.push_back(p);
    p->result_uses.push_back(param(idx));
}
bool 
Op::set_calculate_derivative(const std::vector<Op*>&l){
    if(m_params.size()==0){
        if(l.end() != std::find(l.begin(),l.end(), this)){
            // I'm in the list of ops w.r.t. which derivative is requested
            this->need_derivative(true);
            return true;
        }
        else{
            this->need_derivative(false);
            return false;
        }
    }
	bool need_calc_derivative = false;
	BOOST_FOREACH(param_t& p, m_params){
		bool derive_wrt_p = false;
		BOOST_FOREACH(Op::result_t& r, p->param_uses){

            boost::shared_ptr<Op> op = r->get_op();
            if(!boost::dynamic_pointer_cast<Sink>(op)) // do not derive past sinks
                derive_wrt_p |= r->get_op()->set_calculate_derivative(l);
		}
		p->need_derivative = derive_wrt_p;
        p->determine_single_results();

		need_calc_derivative |= derive_wrt_p;
	}

    // assumes this has been set to false initially!
	m_need_derivative |= need_calc_derivative; 

	return need_calc_derivative;
}

void Op::_determine_shapes(){
	assert(m_params.size()==1);
    assert(m_params[0]->shape.size()>0);
	BOOST_FOREACH(result_t& r, m_results){
		r->shape = m_params[0]->shape;
	}
}
