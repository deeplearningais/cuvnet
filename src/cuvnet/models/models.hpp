#ifndef __CUVNET_MODELS_HPP__
#     define __CUVNET_MODELS_HPP__

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
                std::vector<Op*> m_inputs;

            public:
                /**
                 * return the parameters wrt which we want to derive.
                 * @param stage the name of the current learning stage
                 * @return empty vector by default.
                 */
                virtual std::vector<Op*> get_params(const std::string& stage = "");

                /**
                 * Register any variables that you want the monitor to watch,
                 * eg for statistics over the epoch, or sinks for postprocessing.
                 *
                 * Does nothing by default.
                 *
                 * @param stage the name of the current learning stage
                 * @param mon the object you should register the watches with
                 */
                virtual void register_watches(monitor& mon, const std::string& stage = "");

                /**
                 * return the inputs of the model.
                 * @param stage the name of the current learning stage
                 * @return empty vector by default
                 */
                virtual std::vector<Op*> get_inputs(const std::string& stage = "");

                /**
                 * return the loss of the model (including regularization).
                 * @param stage the name of the current learning stage
                 * @return NULL by default
                 */
                virtual op_ptr loss(const std::string& stage = "")const;

                /**
                 * return the error of the model (eg zero-one loss).
                 * @param stage the name of the current learning stage
                 * @return NULL by default
                 */
                virtual op_ptr error(const std::string& stage = "")const;

                /**
                 * reset all parameters. Does nothing by default.
                 *
                 * @param stage the name of the current learning stage
                 */
                virtual void reset_params(const std::string& stage = "");

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
                        ar & boost::serialization::base_object<model>(*this);
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
                virtual void reset_params(const std::string& stage = "");

                /** 
                 * accumulate params of all submodels.
                 * @param stage the name of the current learning stage
                 */
                virtual std::vector<Op*> get_params(const std::string& stage = "");

                /**
                 * returns the last non-NULL loss from the models, NULL if none found.
                 * @param stage the name of the current learning stage
                 */
                virtual op_ptr loss(const std::string& stage = "")const;

                /**
                 * returns the last non-NULL error from the models, NULL if none found.
                 * @param stage the name of the current learning stage
                 */
                virtual op_ptr error(const std::string& stage = "")const;

                /**
                 * dtor.
                 */
                virtual ~metamodel();
        };
    }
}

#endif /* __CUVNET_MODELS_HPP__ */
