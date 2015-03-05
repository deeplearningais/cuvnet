#ifndef __BOUNDING_BOX_MATCHING_OP_HPP__
#   define __BOUNDING_BOX_MATCHING_OP_HPP__

#include <cuvnet/op.hpp>
#include <cuvnet/datasets/detection.hpp>

namespace cuvnet
{

    class BoundingBoxMatching
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                unsigned int m_n_predictions;
                unsigned int m_n_klasses;
                std::vector<datasets::rotated_rect> m_typical_bboxes;
                std::vector<std::vector<int> > m_matching; 

                cuv::tensor<float, cuv::host_memory_space> prediction;

                std::vector<std::vector<datasets::bbox> > m_teach;
                std::vector<std::vector<datasets::bbox> > m_output_bbox;

                float m_f_match;
                float m_f_conf;
                float m_loss;
                /// controls the ratio of both errors in the final loss
                float m_alpha;

                std::vector<std::vector<datasets::rotated_rect> > m_delta_matching;
                std::vector<std::vector<float> > m_delta_conf;

                std::pair<float, float> loss_terms();

                /**
                 * the washington dataset rendered by Ishrat Badami contains
                 * only one object in its original position, everything else is
                 * clutter, objects of a different class.
                 * 
                 * This parameter /ignores/ all detections which are not of the
                 * class of the first object.
                 */
                bool m_first_object_class_only;

            public:
                BoundingBoxMatching()
                    : m_n_klasses(1)
                    , m_first_object_class_only(false)
                {} ///< for serialization
                /**
                 * ctor.
                 * @param p0 bounding box predictions (offsets) with B x (K * N * 5) dimensions
                 *        where B is batch size, K is number of classes, N is
                 *        number of bounding boxes, and the 5 bounding box
                 *        parameters are center offset, height/width offset,
                 *        and confidence.
                 */
                BoundingBoxMatching(result_t& p0, std::vector<datasets::rotated_rect> kmeans, float alpha, int n_klasses)
                    : Op(1,1)
                    , m_n_klasses(n_klasses)
                    , m_typical_bboxes(kmeans)
                    , m_alpha(alpha)
                    , m_first_object_class_only(false)
                {
                    add_param(0,p0);
                }
                void fprop();
                void bprop();
                void _determine_shapes();

                std::vector<std::vector<datasets::bbox> > get_output_bbox(){ return m_output_bbox; };
                float get_f_match(){ return m_f_match; };
                float get_f_conf(){ return m_f_conf; };

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_n_predictions;
                        ar & m_alpha;
                        ar & m_typical_bboxes; 
                        if(version > 0)
                            ar & m_n_klasses;
                    }
        };
}
BOOST_CLASS_VERSION(cuvnet::BoundingBoxMatching, 1)

#endif // __BOUNDING_BOX_MATCHING_OP_HPP__
