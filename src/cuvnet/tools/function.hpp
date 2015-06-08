#ifndef __CUVNET_FUNCTION_HPP__
#     define __CUVNET_FUNCTION_HPP__


#include <cuvnet/op.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/tools/serialization_helper.hpp>
#include <boost/serialization/split_member.hpp>

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

            template<class Archive>
            void save(Archive & ar, const unsigned int version) const
            {
                ar << m_op << m_sink;
            }
            template<class Archive>
            void load(Archive & ar, const unsigned int version) 
            {
                ar >> m_op >> m_sink;
                m_swiper.reset(new swiper(*m_op, 0, std::vector<Op*>()));
            }
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) { 
                    boost::serialization::split_member(ar, *this, version);
                }

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
             * Add an output to the function
             * @param op the other output requested
             * @param result the index of the op's result we're interested 
             */
            void add(boost::shared_ptr<Op> op, int result=0){
                m_swiper->request_other_result(*op, result);
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
            const matrix& evaluate(bool reinitialize=true){
                if(reinitialize)
                    repair_swiper();
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
            /**
             * Clear all data in graph.
             */
            void forget(){
                m_sink->forget();
                cleanup_temp_vars_visitor ctvv;
                m_op->visit(ctvv,true);
            }

            /**
             * tells swiper not to clean up temporary variables. This is useful
             * when the function object is destroyed before all variables can
             * be collected. A typical use case is in the Op.evaluate()
             * function exported to python, which returns the main result, but
             * not the "additional" results requested. It then destroys the
             * function object, removing the intermediate results. To allow the
             * user access to the results she requested, the Op.evaluate() function
             * requests temporary variables to be kept using this function.
             */
            inline void set_cleanup_temp_vars(bool b){
                m_swiper->set_cleanup_temp_vars(b);
            }
    };

    /**
     * a function that wraps an \c Op, can be evaluated and the partial derivative wrt a specific op accessed.
     *
     * @ingroup tools
     */
    class delta_function{
        private:
            boost::shared_ptr<Op>   m_loss;  // the function to be derived
            boost::shared_ptr<Op>   m_op;    // the parameter which we derive for
            boost::shared_ptr<DeltaSink> m_sink;
            boost::scoped_ptr<swiper>   m_swiper;
        public:

            template<class Archive>
            void save(Archive & ar, const unsigned int version) const
            {
                ar << m_loss << m_op << m_sink;
            }
            template<class Archive>
            void load(Archive & ar, const unsigned int version) 
            {
                ar >> m_loss >> m_op >> m_sink;
                m_swiper.reset(new swiper(*m_loss, 0, std::vector<Op*>(1, m_op.get())));
            }
            friend class boost::serialization::access;
            template<class Archive>
                void serialize(Archive& ar, const unsigned int version) { 
                    boost::serialization::split_member(ar, *this, version);
                }

            delta_function(){}
            inline void repair_swiper(){
                m_swiper->init();
            }
            /**
             * create a function object.
             * @param loss the wrapped op to be derived
             * @param op the wrapped op to be derived for
             * @param result the index of the op's result we're interested in
             * @param param the index of the op's parameter we're interested in
             * @param name a name for the sink (for visualization purposes only)
             */
            delta_function(boost::shared_ptr<Op> loss, boost::shared_ptr<Op> op, int result=0, int param=0, const std::string& name=""){
                reset(loss, op, result, param, name);
            }

            /**
             * re-create a function object.
             * @param loss the wrapped op to be derived
             * @param op the wrapped op to be derived for
             * @param result the index of the loss's result we're interested in
             * @param param the index of the op's param we're interested in
             * @param name a name for the sink (for visualization purposes only)
             */
            void reset(boost::shared_ptr<Op> loss, boost::shared_ptr<Op> op, int result=0, int param=0, const std::string& name=""){
                m_loss = loss;
                m_op   = op;
                m_sink = delta_sink(name, op, param);
                m_swiper.reset(new swiper(*m_loss, result, std::vector<Op*>(1, m_sink.get())));
            }
            
            /**
             * run the function using current inputs
             *
             * @return the value returned by the function
             */
            const matrix& evaluate(bool reinitialize=true){
                if(reinitialize)
                    repair_swiper();
                //m_swiper->dump("delta_function.dot", true);
                m_swiper->fprop();
                m_swiper->bprop();
                return m_sink->cdata();
            }
            /**
             * Access the result after evaluate ran.
             * @return the value returned by the function
             */
            const matrix& result()const{
                return m_sink->cdata();
            }

            void forget(){
                m_sink->forget();
                cleanup_temp_vars_visitor ctvv;
                m_loss->visit(ctvv,true);
            }

            /**
             * tells swiper not to clean up temporary variables. This is useful
             * when the function object is destroyed before all variables can
             * be collected. A typical use case is in the Op.evaluate()
             * function exported to python, which returns the main result, but
             * not the "additional" results requested. It then destroys the
             * function object, removing the intermediate results. To allow the
             * user access to the results she requested, the Op.evaluate() function
             * requests temporary variables to be kept using this function.
             */
            inline void set_cleanup_temp_vars(bool b){
                m_swiper->set_cleanup_temp_vars(b);
            }
    };
}

#endif /* __CUVNET_FUNCTION_HPP__ */
