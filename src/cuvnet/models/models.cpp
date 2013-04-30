#include "models.hpp"

namespace cuvnet { namespace models {

    std::vector<Op*> model::get_params(){
        return std::vector<Op*>();
    }

    std::map<unsigned int, Op*> model::get_inputs(){
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

    model::~model(){}

    /****************************************
     *
     *  metamodel
     *
     ****************************************/

    void metamodel::register_submodel(model& m){
        m_models.push_back(&m);
    }
    void metamodel::reset_params(){
        std::for_each(
                m_models.begin(), m_models.end(), std::mem_fun(&model::reset_params));
    }

    std::vector<Op*> metamodel::get_params(){
        std::vector<Op*> params;

        for(std::vector<model*>::iterator it = m_models.begin(); it!=m_models.end(); it++){
            std::vector<Op*> p = (*it)->get_params();
            std::copy(p.begin(), p.end(), std::back_inserter(params));
        }

        return params;
    }

    metamodel::op_ptr metamodel::loss()const{
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++){
            op_ptr l = (*it)->loss();
            if(l)
                return l;
        }
        return op_ptr();
    }
    metamodel::op_ptr metamodel::error()const{
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++){
            op_ptr l = (*it)->error();
            if(l)
                return l;
        }
        return op_ptr();
    }
    metamodel::~metamodel(){}
        
} }
