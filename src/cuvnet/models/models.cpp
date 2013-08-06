#include "models.hpp"

namespace cuvnet { namespace models {

    std::vector<Op*> model::get_params(const std::string& stage){
        return std::vector<Op*>();
    }

    std::vector<Op*> model::get_inputs(const std::string& stage){
        return m_inputs;
    }

    model::op_ptr model::loss(const std::string& stage)const{
        return op_ptr();
    }

    model::op_ptr model::error(const std::string& stage)const{
        return op_ptr();
    }

    void model::reset_params(const std::string& stage){
    }

    void model::register_watches(monitor& mon, const std::string& stage){
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
    void metamodel::reset_params(const std::string& stage){
        // use bind2nd, since the 1st param of mem_fun is the implicit `this'
        // http://stackoverflow.com/questions/1762781/mem-fun-and-bind1st-problem
        std::for_each(
                m_models.begin(), m_models.end(), std::bind2nd(std::mem_fun(&model::reset_params), stage));
    }

    std::vector<Op*> metamodel::get_params(const std::string& stage){
        std::vector<Op*> params;

        for(std::vector<model*>::iterator it = m_models.begin(); it!=m_models.end(); it++){
            std::vector<Op*> p = (*it)->get_params(stage);
            std::copy(p.begin(), p.end(), std::back_inserter(params));
        }

        return params;
    }

    metamodel::op_ptr metamodel::loss(const std::string& stage)const{
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++){
            op_ptr l = (*it)->loss(stage);
            if(l)
                return l;
        }
        return op_ptr();
    }
    metamodel::op_ptr metamodel::error(const std::string& stage)const{
        for(std::vector<model*>::const_reverse_iterator it = m_models.rbegin(); it!=m_models.rend(); it++){
            op_ptr l = (*it)->error(stage);
            if(l)
                return l;
        }
        return op_ptr();
    }
    metamodel::~metamodel(){}
        
} }
