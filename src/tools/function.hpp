#ifndef __CUVNET_FUNCTION_HPP__
#     define __CUVNET_FUNCTION_HPP__


#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops/output.hpp>

namespace cuvnet
{
    class function{
        private:
            boost::shared_ptr<Op>   m_op;
            boost::shared_ptr<Sink> m_sink;
            swiper                  m_swiper;
        public:
            function(boost::shared_ptr<Op> op, int result=0, const std::string& name="")
                : m_op(op)
                , m_sink(boost::make_shared<Sink>(name, m_op->result(result)))
                , m_swiper(*m_op, result, std::vector<Op*>()){
                }
            
            const matrix& evaluate(){
                m_swiper.fprop();
                return m_sink->cdata();
            }
            const matrix& result()const{
                return m_sink->cdata();
            }
    };
}

#endif /* __CUVNET_FUNCTION_HPP__ */