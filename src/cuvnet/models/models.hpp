#ifndef __CUVNET_MODELS_HPP__
#     define __CUVNET_MODELS_HPP__

#include <map>
#include <vector>
#include <cuvnet/op.hpp>

namespace cuvnet
{
    class monitor;

    namespace models
    {
        struct model{
            private:
                typedef boost::shared_ptr<Op> op_ptr;

                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        //ar & boost::serialization::base_object<boost::enable_shared_from_this<Op> >(*this);
                        ar & m_inputs;
                    }

            protected:
                std::map<unsigned int, Op*> m_inputs;

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
                 * dtor.
                 */
                virtual ~model();
        };


        /**
         * A metamodel dispatches calls to the standard functions to all
         * submodels.
         */
        struct metamodel : public model{
            private:
                typedef boost::shared_ptr<Op> op_ptr;

                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<boost::enable_shared_from_this<model> >(*this);
                        ar & m_models;
                    }
            protected:
                std::vector<model*> m_models;

                /** 
                 * add a model to the list of submodels.
                 * @param m  the model to be added
                 */
                void register_submodel(model& m);

            public:
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
                 * dtor.
                 */
                virtual ~metamodel();
        };
    }
}

#endif /* __CUVNET_MODELS_HPP__ */
