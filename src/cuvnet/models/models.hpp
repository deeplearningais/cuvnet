#ifndef __CUVNET_MODELS_HPP__
#     define __CUVNET_MODELS_HPP__

#include <vector>
#include <cuvnet/op.hpp>
#include <boost/serialization/export.hpp>                                                                                                        

namespace cuvnet
{
    class monitor;

    namespace models
    {
        struct model
        {
            private:
                typedef boost::shared_ptr<Op> op_ptr;

                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        //ar & boost::serialization::base_object<boost::enable_shared_from_this<Op> >(*this);
                        ar & m_inputs;
                    }

            protected:
                std::vector<Op*> m_inputs;

            public:
                /**
                 * return the parameters wrt which we want to derive.
                 * @return empty vector by default.
                 */
                virtual std::vector<Op*> get_params();

                /**
                 * Register any variables that you want the monitor to watch,
                 * eg for statistics over the epoch, or sinks for postprocessing.
                 *
                 * Does nothing by default.
                 *
                 * @param mon the object you should register the watches with
                 */
                virtual void register_watches(monitor& mon);

                /**
                 * return the inputs of the model.
                 * @return empty vector by default
                 */
                virtual std::vector<Op*> get_inputs();

                /**
                 * return the loss of the model (including regularization).
                 * @return NULL by default
                 */
                virtual op_ptr loss()const;

                /**
                 * return the error of the model (eg zero-one loss).
                 * @return NULL by default
                 */
                virtual op_ptr error()const;

                /**
                 * reset all parameters. Does nothing by default.
                 */
                virtual void reset_params();

                /**
                 * Should turn off all noise introduced during training, eg by Dropout. Does nothing by default.
                 */
                virtual void set_predict_mode(bool b=true);

                /**
                 * Change the batch size of the model. Changes the first
                 * dimension of all params returned by get_inputs().
                 */
                virtual void set_batchsize(unsigned int bs);

                /**
                 * Run after weight update, eg to project weights onto a legit region.
                 * Does nothing by default.
                 */
                virtual void after_weight_update();

                /**
                 * dtor.
                 */
                virtual ~model();
        };

        struct multistage_model : public model{
            public:
                typedef unsigned int stage_type;
                typedef boost::shared_ptr<Op> op_ptr;

            private:
                unsigned int m_n_stages;
                stage_type m_current_stage;

                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<model>(*this);
                        ar & m_current_stage;
                        //cuvAssert(m_current_stage == 0);
                    }
            public:

                /**
                 * ctor.
                 * @param n_stages the number of stages in this model
                 */
                multistage_model(unsigned int n_stages = 1)
                    : m_n_stages(n_stages)
                      , m_current_stage(0){}
                /**
                 * return the number of stages.
                 *
                 * @return 1
                 */
                inline unsigned int n_stages()const{return m_n_stages;}

                /**
                 * return the stage the model is currently in.
                 */
                inline const stage_type& current_stage()const{ return m_current_stage; }

                /**
                 * switch to another stage.
                 * @param stage the stage to switch to
                 */
                virtual void switch_stage(const stage_type& stage);

                /**
                 * get the outputs of the currently active stage.
                 * @return empty vector.
                 */
                virtual std::vector<Op*> get_outputs();
        };

        /**
         * A metamodel dispatches calls to the standard functions to all
         * submodels.
         *
         * @tparam Base either model or multistage_model
         */
        template<class Base>
        struct metamodel : public Base {
            private:
                typedef boost::shared_ptr<Op> op_ptr;
                typedef Base base_t;

                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Base>(*this);
                        ar & m_models;
                    }

                void  _reset_params(metamodel<multistage_model>* p);
                void  _reset_params(model* p);

            protected:
                std::vector<model*> m_models;

            public:
                /** 
                 * add a model to the list of submodels.
                 * @param m  the model to be added
                 */
                void register_submodel(model& m);

                /**
                 * remove a submodel from the list of submodels.
                 * @param m the submodel to be removed
                 */
                void deregister_submodel(model& m);

                /**
                 * clear list of submodels.
                 */
                void clear_submodels();

            public:
                template<class T>
                metamodel(const T& t)
                : Base(t){};

                metamodel(){};

                /**
                 * reset all submodels.
                 */
                virtual void reset_params();

                /** 
                 * accumulate params of all submodels.
                 */
                virtual std::vector<Op*> get_params();

                /**
                 * returns the last non-NULL loss from the models, NULL if none found.
                 */
                virtual op_ptr loss()const;

                /**
                 * returns the last non-NULL error from the models, NULL if none found.
                 */
                virtual op_ptr error()const;

                /**
                 * calls register_watches for all submodels.
                 * @param mon this parameter is passed through.
                 */
                virtual void register_watches(monitor& mon);

                /**
                 * Calls set_predict_mode for all submodels.
                 */
                virtual void set_predict_mode(bool b=true);

                /**
                 * Run after weight update for all submodels.
                 */
                virtual void after_weight_update();

                /**
                 * dtor.
                 */
                virtual ~metamodel();
        };
    }
}
BOOST_CLASS_EXPORT_KEY(cuvnet::models::model) 
BOOST_CLASS_EXPORT_KEY(cuvnet::models::multistage_model) 
BOOST_CLASS_EXPORT_KEY(cuvnet::models::metamodel<cuvnet::models::model>) 
BOOST_CLASS_EXPORT_KEY(cuvnet::models::metamodel<cuvnet::models::multistage_model>) 

#endif /* __CUVNET_MODELS_HPP__ */
