#include "models.hpp"
#include <cuvnet/ops/input.hpp>

namespace cuvnet { namespace models {

    std::vector<Op*> model::get_params(){
        return std::vector<Op*>();
    }

    std::vector<Op*> model::get_inputs(){
        return m_inputs;
    }

    model::op_ptr model::loss()const{
        return op_ptr();
    }

    model::op_ptr model::error()const{
        return op_ptr();
    }

    void model::reset_params(){
    }

    void model::register_watches(monitor& mon){
    }
    void model::set_predict_mode(bool b){
    }
    void model::after_weight_update(){
    }

    void model::set_batchsize(unsigned int bs){
        std::vector<Op*> inputs = get_inputs();
        for(unsigned int i=0; i<inputs.size(); i++){
            ParameterInput* inp = (ParameterInput*) inputs[i];
            std::vector<unsigned int> shape = inp->data().shape();
            if(shape.size() == 0)
                continue;
            shape[0] = bs;
            inp->data().resize(shape);
        }
    }

    model::~model(){}

    /****************************************
     *
     *  multistage_model
     *
     ****************************************/

    void multistage_model::switch_stage(const stage_type& stage){
        m_current_stage = stage;
    }

    std::vector<Op*>
        multistage_model::get_outputs(){
            return std::vector<Op*>();
        }

    /****************************************
     *
     *  metamodel
     *
     ****************************************/

    template<class Base>
    void metamodel<Base>::register_submodel(model& m){
        m_models.push_back(&m);
    }

    template<class Base>
    void metamodel<Base>::deregister_submodel(model& m){
        m_models.erase(std::remove(m_models.begin(), m_models.end(), &m),
                m_models.end());
    }

    template<class Base>
    void metamodel<Base>::clear_submodels(){
        m_models.clear();
    }

    template<class Base>
    void
    metamodel<Base>::reset_params(){
        _reset_params(this);
    }

    template<class Base>
    void
    metamodel<Base>::_reset_params(metamodel<multistage_model>* p){
        for(unsigned int stage = 0; stage < p->n_stages(); stage ++){
            p->switch_stage(stage);
            std::for_each(
                m_models.begin(), m_models.end(), std::mem_fun(&model::reset_params));
        }
    }
                
    template<class Base>
    void 
    metamodel<Base>::_reset_params(model* p){
        std::for_each(
            m_models.begin(), m_models.end(), std::mem_fun(&model::reset_params));
    }

    template<class Base>
    std::vector<Op*> metamodel<Base>::get_params(){
        std::vector<Op*> params;

        for(std::vector<model*>::iterator it = m_models.begin(); it!=m_models.end(); it++){
            std::vector<Op*> p = (*it)->get_params();
            std::copy(p.begin(), p.end(), std::back_inserter(params));
        }

        return params;
    }

    template<class Base>
    typename metamodel<Base>::op_ptr 
    metamodel<Base>::loss()const{
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++){
            op_ptr l = (*it)->loss();
            if(l)
                return l;
        }
        return op_ptr();
    }

    template<class Base>
    typename metamodel<Base>::op_ptr 
    metamodel<Base>::error()const{
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++){
            op_ptr l = (*it)->error();
            if(l)
                return l;
        }
        return op_ptr();
    }

    template<class Base>
    void
    metamodel<Base>::register_watches(monitor& mon){
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++)
            (*it)->register_watches(mon);
    }

    template<class Base>
    void
    metamodel<Base>::set_predict_mode(bool b){
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++)
            (*it)->set_predict_mode(b);
    }

    template<class Base>
    void
    metamodel<Base>::after_weight_update(){
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++)
            (*it)->after_weight_update();
    }

    template<class Base>
    metamodel<Base>::~metamodel(){}
        
    template class metamodel<model>;
    template class metamodel<multistage_model>;
} }


BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::model);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::multistage_model);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::metamodel<cuvnet::models::model>);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::metamodel<cuvnet::models::multistage_model>);
