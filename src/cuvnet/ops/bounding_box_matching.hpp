#ifndef __BOUNDING_BOX_MATCHING_OP_HPP__
#   define BOUNDING_BOX_MATCHING_OP_HPP__

#include <cuvnet/op.hpp>

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

                struct bbox {
                    float x_min, y_min, x_max, y_max;
                    bbox():x_min(0),y_min(0),x_max(0),y_max(0){};
                    bbox(float x_min, float y_min, float x_max, float y_max)
                        : x_min(x_min)
                        , y_min(y_min)
                        , x_max(x_max)
                        , y_max(y_max) {};
                    static inline float l2dist(bbox a, bbox b) {
                        return std::sqrt(
                                std::pow(a.x_min - b.x_min, 2) + 
                                std::pow(a.y_min - b.y_min, 2) +
                                std::pow(a.x_max - b.x_max, 2) +
                                std::pow(a.y_max - b.y_max, 2)
                                );
                    };

                    friend bbox operator+(const bbox& l, const bbox& r){
                        return bbox(
                                l.x_min + r.x_min,
                                l.y_min + r.y_min,
                                l.x_max + r.x_max,
                                l.y_max + r.y_max
                                );
                    };
                    friend bbox operator-(const bbox& l, const bbox& r){
                        return bbox(
                                l.x_min - r.x_min,
                                l.y_min - r.y_min,
                                l.x_max - r.x_max,
                                l.y_max - r.y_max
                                );
                    };
                    inline bbox& scale_like_vec(float f){
                        x_min *= f;
                        y_min *= f;
                        x_max *= f;
                        y_max *= f;
                        return *this;
                    };
                };

            private:
                unsigned int m_K;
                std::vector<bbox> m_typical_bboxes;
                std::vector<std::vector<int> > m_matching; 

                std::vector<std::vector<bbox> > m_teach_bbox;
                std::vector<std::vector<bbox> > m_output_bbox;
                std::vector<std::vector<float> > m_output_conf;

                float m_f_match;
                float m_f_conf;

                /// controls the ratio of both errors in the final loss
                float m_alpha;


                std::pair<float, float> loss_terms();

            public:
                BoundingBoxMatching(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 bounding box predictions (offsets) with B x N x 4 dimenions
                 * @param p1 network confidence of each bounding box B X N dimensions
                 */
                BoundingBoxMatching(result_t& p0, std::vector<bbox> kmeans, float alpha)
                    : Op(1,1)
                    , m_typical_bboxes(kmeans)
                    , m_alpha(alpha){
                    add_param(0,p0);
                    //add_param(1,p1);
                    m_results[0]->delta           = value_ptr(new value_type(cuv::extents[1], value_ptr::s_allocator));
                    m_results[0]->delta.data()[0] = 1.f;
                }
                void fprop();
                void bprop();
                void _determine_shapes();

                std::vector<std::vector<bbox> > get_output_bbox(){ return m_output_bbox; };
                void set_teacher_bbox( std::vector<std::vector<bbox> > teach ){ m_teach_bbox = teach; };
                float get_f_match(){ return m_f_match; };
                float get_f_conf(){ return m_f_conf; };

            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        //ar & m_identity;
                    }
        };
}

#endif // __BOUNDING_BOX_MATCHING_OP_HPP__
