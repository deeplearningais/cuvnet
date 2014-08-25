#ifndef __OP_UPSCALE_HPP__
#     define __OP_UPSCALE_HPP__

#include <cuvnet/op.hpp>
#include <boost/limits.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
namespace cuvnet
{       
     /**
     * Upscale image
     *      
     * 
     * @ingroup Ops
     */
    class Upscale
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
            private:
                int factor;
            public:
                Upscale() :Op(1,1){} ///< for serialization
                /**
                 * ctor.
                 * @param in the input
                 * @param eg the extents after reshaping.
                 */
                Upscale(result_t& in, unsigned int fact)
                    :Op(1,1),factor(fact)
                {
                    //add param
                    add_param(0, in);
                }
                void fprop();
                void bprop();

                void _determine_shapes();
                value_type get_subtensor(const value_type &v, unsigned int position);
            protected:
                    boost::shared_ptr<std::vector<unsigned int> > get_pi_shape(value_type & vi);
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & factor;
                    }
        };
}

#endif /* __OP_UPSCALE_HPP__ */
