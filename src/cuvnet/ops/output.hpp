#ifndef __OP_OUTPUT_HPP__
#     define __OP_OUTPUT_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * A sink can be used to reuse an intermediate value *after* a function has been evaluated.
     *
     * There are two main use-cases for this:
     * - You want to access an intermediate result of a function and do not
     *   want it to be overwritten or deleted for space-optimization.
     * - You want to access the result of a function and do not want it to be
     *   overwritten or deleted for space-optimization during backpro pagation.
     * 
     * A \c Sink has one output. This enables it to work like an \c Input to
     * another function. Thus, you can evaluate a second function based on the
     * intermediate result of a first function. An example is 
     * \c logistic_regression, where the output of the predictor is used for
     * the logistic loss *and* for the classification loss.
     *
     * @ingroup Ops
     */
    class Sink
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                Sink(){} ///< for serialization
                /**
                 * ctor with name.
                 * @param name the name of this sink for visualization
                 * @param p0 the input
                 */
                Sink(const std::string& name, result_t& p0):Op(1,1),m_name(name){ 
                    add_param(0,p0);
                }
                /**
                 * ctor without name.
                 * @param p0 the input
                 */
                Sink(result_t& p0):Op(1,1){ 
                    add_param(0,p0);
                }
                ///  A sink always pretends it wants the result
                /// @overload
                virtual bool need_result()const{return true;}
                void fprop();
                void bprop();
                //void _determine_shapes(){ }
                //value_type&       data()      { return m_data; }
                /// @return a (constant) reference to the stored data
                const value_type& cdata() const{ return m_params[0]->value.cdata(); }

                /**
                 * manually forget the stored value.
                 *
                 * This save space during bprop, as it can be overwritten if only
                 * a single copy remains now.
                 */
                void forget();

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                /// @return the name of this input
                inline const std::string& name()const{ return m_name; }
            private:
                std::string    m_name;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_name;
                    }
        };

    /**
     * A delta sink is like a Sink, but it can be attached to op_params to save
     * the result of a delta calculation.
     *
     * @ingroup Ops
     */
    class DeltaSink
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            public:
                DeltaSink(){} ///< for serialization
                /**
                 * ctor.
                 * @param name the name of this delta sink.
                 */
                DeltaSink(const std::string& name):Op(0,1),m_name(name){ 
                }
                ///  A sink always pretends it wants the derivative
                /// @overload
                virtual bool need_derivative()const{return true;}
                void fprop();
                void bprop();
                /// @return a (constant) reference to the stored data
                inline const value_type& cdata() const{ return m_results[0]->delta.cdata(); }

                /**
                 * forget the stored value.
                 */
                void forget();

                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
            private:
                std::string    m_name;
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_name;
                    }
        };

    /**
     * Convenience op, has the \f$n\f$-th result of its input as its (only) output.
     *
     * This makes it more convenient to pass the e.g. second output of an op on
     * to another function, since most functions assume that the input has only
     * one output.
     *
     * @ingroup Ops
     */
    class Pipe
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;
                unsigned int m_idx; ///< needed for visualization only

            public:
                Pipe(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 the input op with possibly many results
                 * @param idx the index of the result to pass through as the first result of this pipe
                 */
                Pipe(result_t& p0, unsigned int idx):Op(1,1), m_idx(idx){ 
                    add_param(0,p0);
                }
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const;
                void fprop();
                void bprop();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_idx;
                    }
        };

}
#endif /* __OP_OUTPUT_HPP__ */
