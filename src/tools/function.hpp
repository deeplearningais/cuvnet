#ifndef __CUVNET_FUNCTION_HPP__
#     define __CUVNET_FUNCTION_HPP__


#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops/output.hpp>

namespace cuvnet
{
    /**
     * a function that wraps an \c Op, can be evaluated and the result accessed.
     *
     * @ingroup tools
     */
    class function{
        private:
            boost::shared_ptr<Op>   m_op;
            boost::shared_ptr<Sink> m_sink;
            std::auto_ptr<swiper>   m_swiper;
        public:
            function(){}
            inline void repair_swiper(){
                m_swiper->init();
            }
            /**
             * create a function object.
             * @param op the wrapped op
             * @param result the index of the op's result we're interested in
             * @param name a name for the sink (for visualization purposes only)
             */
            function(boost::shared_ptr<Op> op, int result=0, const std::string& name=""){
                reset(op,result,name);
            }

            /**
             * re-create a function object.
             * @param op the wrapped op
             * @param result the index of the op's result we're interested in
             * @param name a name for the sink (for visualization purposes only)
             */
            void reset(boost::shared_ptr<Op> op, int result=0, const std::string& name=""){
                m_op   = op;
                m_sink = boost::make_shared<Sink>(name, op->result(result));
                m_swiper.reset(new swiper(*m_op, result, std::vector<Op*>()));
            }
            
            /**
             * run the function using current inputs
             *
             * @return the value returned by the function
             */
            const matrix& evaluate(){
                m_swiper->fprop();
                return m_sink->cdata();
            }
            /**
             * Access the result after evaluate ran.
             * @return the value returned by the function
             */
            const matrix& result()const{
                return m_sink->cdata();
            }
    };
}

#endif /* __CUVNET_FUNCTION_HPP__ */
