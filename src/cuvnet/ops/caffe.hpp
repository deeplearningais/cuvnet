
#ifndef CAFFE_HPP_
#define CAFFE_HPP_

#include <cuv/libs/caffe/caffe.hpp>
#include <cuvnet/op.hpp>

namespace cuvnet
{
/**
  * Caffe Response Norm across maps.
  *
  * @ingroup CaffeOps
  */
 class ResponseNormalizationAcrossMapsCaffe
     : public Op{
         public:
             typedef Op::value_type    value_type;
             typedef Op::op_ptr        op_ptr;
             typedef Op::value_ptr     value_ptr;
             typedef Op::param_t       param_t;
             typedef Op::result_t      result_t;
         private:
             int m_group_size;
             float m_add_scale, m_pow_scale;
             value_ptr m_orig_out;
             matrix m_denom;
         public:
             ResponseNormalizationAcrossMapsCaffe() :Op(1,1){} ///< for serialization

             /**
              * ctor.
              * @param images input image function
              * @param group_size number of maps around center to use for normalization
              * @param add_scale add this to denominator
              * @param pow_scale raise denominator to this power
              */
             ResponseNormalizationAcrossMapsCaffe(result_t& images, int group_size, float add_scale, float pow_scale)
                 :Op(1,1),
                 m_group_size(group_size),
                 m_add_scale(add_scale),
                 m_pow_scale(pow_scale)
             {
                 add_param(0,images);
             }
             virtual void release_data();
             void fprop();
             void bprop();

             void _determine_shapes();

         private:
             friend class boost::serialization::access;
             template<class Archive>
                 void serialize(Archive& ar, const unsigned int version){
                     ar & boost::serialization::base_object<Op>(*this);
                     ar & m_group_size & m_add_scale & m_pow_scale;
                 }
     };

}



#endif /* CAFFE_HPP_ */
