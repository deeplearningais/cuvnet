#include "models.hpp"

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
    void metamodel<Base>::reset_params(){
        // use bind2nd, since the 1st param of mem_fun is the implicit `this'
        // http://stackoverflow.com/questions/1762781/mem-fun-and-bind1st-problem
        std::for_each(
                //m_models.begin(), m_models.end(), std::bind2nd(std::mem_fun(&model::reset_params), stage));
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
    metamodel<Base>::~metamodel(){}
        
    template class metamodel<model>;
    template class metamodel<multistage_model>;
} }


BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::model);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::multistage_model);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::metamodel<cuvnet::models::model>);
BOOST_CLASS_EXPORT_IMPLEMENT(cuvnet::models::metamodel<cuvnet::models::multistage_model>);
