#ifndef INCEPTION_HPP__38264823
#define INCEPTION_HPP__38264823

#include <boost/tuple/tuple.hpp>
#include <cuvnet/tools/monitor.hpp>
#include <cuvnet/models/models.hpp>
#include <cuvnet/models/conv_layer.hpp>
#include <cuvnet/models/mlp.hpp>

namespace cuvnet { namespace models {

    struct inception_layer : public metamodel<model>{
        public:
            typedef boost::shared_ptr<Op> op_ptr;
            typedef boost::shared_ptr<ParameterInput> input_ptr;
            typedef boost::shared_ptr<Sink> sink_ptr;
            std::vector<boost::shared_ptr<conv_layer> > m_convlayers;
            std::vector<boost::shared_ptr<mlp_layer> > m_fclayers;
            std::vector<input_ptr> m_weights;
            op_ptr m_input;
            op_ptr m_output;

            /**
             * @param m filter sizes, and to what dimension they are to be compressed before convolution.
             *          - (1,  X, 64) means that the input is compressed to 64 dimensions by a 1x1 convolution
             *          - (3, 64, 28) means that the input is compressed to 64 dimensions, then convolved by a 3x3 convolution to 28 maps
             *          - (-1, 3,  X) means that the input is ran through a max-pooling with stride=poolsize=3
             *          X denote values which are ignored.
             */
            inception_layer(op_ptr input, const std::vector<boost::tuple<int,int,int> >& m);

            /**
             * standard configuration constructor (1x1, 3x3, 5x5 and max-pooling layer)
             *
             * @param input source layer
             * @param n_dst_maps number of destination maps for each of the components
             */
            inception_layer(op_ptr input, int n_dst_maps);

            /**
             * default ctor, for serialization.
             */
            inception_layer(){}

        private:
            void init(op_ptr input, const std::vector<boost::tuple<int,int,int> >& m);
            friend class boost::serialization::access;
            template <class Archive>
                void serialize(Archive& ar, const unsigned int){
                    ar & boost::serialization::base_object<metamodel<model> >(*this);
                    ar & m_convlayers& m_fclayers;
                    ar & m_weights & m_input & m_output;
                }

    };

}}
BOOST_CLASS_EXPORT_KEY(cuvnet::models::inception_layer);
#endif /* INCEPTION_HPP__38264823 */
