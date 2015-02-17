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
                unsigned int m_K;
                std::vector<datasets::rotated_rect> m_typical_bboxes;
                std::vector<std::vector<int> > m_matching; 

                //std::vector<std::vector<datasets::rotated_rect> > m_teach_bbox;
                std::vector<std::vector<datasets::bbox> > m_teach;
                std::vector<std::vector<datasets::rotated_rect> > m_output_bbox;
                std::vector<std::vector<float> > m_output_conf;

                float m_f_match;
                float m_f_conf;
                float m_loss;
                /// controls the ratio of both errors in the final loss
                float m_alpha;

                std::vector<std::vector<datasets::rotated_rect> > m_delta_matching;
                std::vector<std::vector<float> > m_delta_conf;

                std::pair<float, float> loss_terms();

            public:
                BoundingBoxMatching(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 bounding box predictions (offsets) with B x N x 4 dimenions
                 * @param p1 network confidence of each bounding box B X N dimensions
                 */
                BoundingBoxMatching(result_t& p0, std::vector<datasets::rotated_rect> kmeans, float alpha)
                    : Op(1,1)
                    , m_typical_bboxes(kmeans)
                    , m_alpha(alpha){
                    add_param(0,p0);
                    m_results[0]->delta           = value_ptr(new value_type(cuv::extents[1], value_ptr::s_allocator));
                    m_results[0]->delta.data()[0] = 1.f;
                }
                void fprop();
                void bprop();
                void _determine_shapes();

                std::vector<std::vector<datasets::rotated_rect> > get_output_bbox(){ return m_output_bbox; };
                //void set_teacher_bbox( std::vector<std::vector<datasets::rotated_rect> > teach ){ m_teach_bbox = teach; };
                //void set_teacher_bbox( std::vector<std::vector<datasets::rotated_rect> > teach ){ m_teach_bbox = teach; };
                float get_f_match(){ return m_f_match; };
                float get_f_conf(){ return m_f_conf; };

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_K;
                        ar & m_alpha;
                        ar & m_typical_bboxes; 
                    }
        };
}

#endif // __BOUNDING_BOX_MATCHING_OP_HPP__
