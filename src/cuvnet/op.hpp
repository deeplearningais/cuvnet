// vim:ts=4:sw=4:et:
#ifndef __OP_HPP__
#     define __OP_HPP__
#include <list>
#include <map>
#include <boost/weak_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/serialization/base_object.hpp>
#include <cuvnet/common.hpp>
#include <cuvnet/smart_ptr.hpp>
#include <cuvnet/detail/op_result.hpp>
#include <cuvnet/detail/op_param.hpp>
#include <cuvnet/detail/graphviz_node.hpp>

namespace cuvnet
{
    /**
     * The central (abstract) symbolic operator all others derive from.
     *
     * An Op has a tuple of parameters and a tuple of results.
     * - In Op::fprop, it calculates the result of the function evaluation.
     * - In Op::bprop, it determines the derivative of the output(s) to all
     *   parameters which need it.
     * - In Op::visit, recursive operations on functions can be performed using
     *   visitors modeling \c op_visitor_adaptor.
     *
     *
     * The \c Op can only exist as a \c boost::shared_ptr<Op>, since it makes
     * use of \c shared_from_this(). The structure is circle-free (using weak
     * pointers in one direction). When the \c Op is destroyed, it
     * automatically detaches from all its results.
     *
     * A note on symbolic differentiation:
     * The Op is completely symmetric regarding forward and backward propagation.
     * In backward propagation the parameters act like the results in forward
     * propagation.
     *
     * So, while each result can be used as a parameter in multiple other \c Ops,
     * likewise, each parameter can be targeted by multiple results of other \c Ops.
     * This always means 'distributing' in one direction and 'adding' in the other.
     *
     * @ingroup Ops
     */
    class Op
        : public boost::enable_shared_from_this<Op>{
            public:
                /// the value type of an Op is -- for now -- always a cuv::tensor.
                typedef matrix value_type;
                /// values are passed around as \c cow_ptr<value_type>.
                typedef cow_ptr<value_type>                 value_ptr;
                /// ops are always stored in \c shared_ptr, so define a shortcut for that
                typedef boost::shared_ptr<Op>               op_ptr;
                /// to break circles, references of results to parameters are stored in \c weak_ptr objects instead of \c shared_ptr
                typedef boost::weak_ptr<detail::op_param<value_type> >     weak_param_t;
                /// parameters stored in Ops are \c shared_ptr
                typedef boost::shared_ptr<detail::op_param<value_type> >        param_t;
                /// results stored in Ops are \c shared_ptr
                typedef boost::shared_ptr<detail::op_result<value_type> >      result_t;
            protected:
                std::vector<param_t>  m_params;  ///< all parameters of the Op
                std::vector<result_t> m_results; ///< all results of the Op
                bool                  m_need_derivative; ///< true if the Op needs to be evaluated in \c fprop
                bool                  m_need_result;  ///< true if the Op needs to be evaluated in \c bprop

            public:
                /**
                 * Default constructor (you shouldn't use this, only implicitly during deserialization!)
                 */
                Op();
                /**
                 * Standard constructor, to be used by derived classes.
                 *
                 * @param n_params number of parameters the Op takes
                 * @param n_results number of results the Op provides
                 */
                Op(unsigned int n_params, unsigned int n_results);

                /**
                 * The destructor detaches the Op from all the elements it was used in.
                 *
                 * That is, if you destroy the last reference you had to the
                 * result of a complex function, the entire function will be
                 * destroyed consecutively.
                 *
                 */
                virtual ~Op();

                Op& detach_from_params(); ///< remove the Op from all its parameters (which then may be destroyed automagically)
                Op& detach_from_results(); ///< remove the Op from all its results (eg to exchange the input of a different Op)

                /** 
                 * returns a reference to the i-th result.
                 *
                 * @note the included check is inefficient but avoids making constructor private
                 *       since we cannot determine `shared_from_this' in
                 *       the constructor when we construct objects in m_results.
                 *       I assume that result() will be primarily used to construct
                 *       functions, which is not that often.
                 */
                result_t&       result(const unsigned int i=0);

                /**
                 * @see result(const unsigned int)
                 */
                result_t& operator[](const unsigned int i){ return result(i); }

                /** 
                 * returns a reference to the i-th parameter
                 */
                param_t&       param(const unsigned int i=0);
                void set_n_params(unsigned int n); ///< set the number of parameters this Op has
                void set_n_results(unsigned int n); ///< set the number of results this Op has
                inline unsigned int get_n_params(){ return m_params.size(); } ///< return the number of parameters this Op has
                inline unsigned int get_n_results(){ return m_results.size(); } ///< return the number of results this Op has
                /**
                 * use the result of another Op as input to this one.
                 * @param idx index of parameter
                 * @param p   result
                 */
                void add_param(unsigned int idx, result_t& p);
                virtual bool need_derivative()const{return m_need_derivative;} ///< return whether Op needs to be called in \c bprop
                inline void need_derivative(bool b){m_need_derivative = b;}    ///< set whether Op needs to be called in \c bprop
                virtual bool need_result()const{return m_need_result;}         ///< return whether Op needs to be called in \c fprop
                inline void need_result(bool b){m_need_result = b;}            ///< set whether Op needs to be called in \c fprop

                /**
                 * Calculate recursively what needs to be calculated to
                 * derive this operator wrt a set of parameters.
                 *
                 * The results are stored in the function itself.
                 *
                 * @param l the list of parameters w.r.t. which this op is to be derived
                 * @return whether this op needs to be calculated.
                 */
                bool set_calculate_derivative(const std::vector<Op*>&l);

                friend struct param_collector_visitor;
                friend struct toposort_visitor;
                friend struct determine_shapes_visitor;
                friend struct cleanup_temp_vars_visitor;
                friend struct reset_value_set_flag;
                friend struct reset_delta_set_flag;
                friend struct define_graphviz_node_visitor;
                friend struct swiper;

                /**
                 * Show all Ops to a (constant) visitor recursively.
                 *
                 * @param v visitor (should model \c op_visitor_adaptor)
                 * @param results_too if true, also visits results. You have to
                 *        do bookkeeping in the visitor to ensure that you are
                 *        not trapped in endless loops.
                 */
                template<class Visitor>
                    void visit(const Visitor& v, bool results_too=false){
                        if(!v.discover(this)) return;
                        v.preorder(this);
                        BOOST_FOREACH(Op::param_t& p, m_params){
                            BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                                r->get_op()->visit(v, results_too);
                            }
                        }
                        if(results_too){
                            BOOST_FOREACH(Op::result_t& p, m_results){
                                BOOST_FOREACH(boost::weak_ptr<detail::op_param<value_type> > r, p->result_uses){
                                    r.lock()->get_op()->visit(v, true);
                                }
                            }
                        }
                        v.postorder(this);
                    }
                /**
                 * Show all Ops to a  visitor recursively.
                 *
                 * @param v visitor (should model \c op_visitor_adaptor)
                 * @param results_too if true, also visits results. You have to
                 *        do bookkeeping in the visitor to ensure that you are
                 *        not trapped in endless loops.
                 */
                template<class Visitor>
                    void visit(Visitor& v, bool results_too=false){
                        if(!v.discover(this)) return;
                        v.preorder(this);
                        BOOST_FOREACH(Op::param_t& p, m_params){
                            BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                                r->get_op()->visit(v,results_too);
                            }
                        }
                        if(results_too){
                            BOOST_FOREACH(Op::result_t& p, m_results){
                                BOOST_FOREACH(boost::weak_ptr<detail::op_param<value_type> > r, p->result_uses){
                                    r.lock()->get_op()->visit(v, results_too);
                                }
                            }
                        }
                        v.postorder(this);
                    }


                /**
                 * user-supplied function: calculate results of this op.
                 */
                virtual void fprop()=0;
                /**
                 * user-supplied function: backpropagate results of this op.
                 */
                virtual void bprop()=0;

                /**
                 * virtual function: determine the shape for each result.
                 *
                 * The default works for ops with only one input:
                 * the shape of the input is simply passed to each result.
                 *
                 * If your Op does more complex things, i.e. is not elementwise, 
                 * you have to overwrite this function.
                 */
                virtual void _determine_shapes();

                /**
                 * modify the graphviz node description string.
                 *
                 * You should mainly set the label of the node here.
                 * The default uses runtime type information to generate a
                 * (more or less ugly) node name.
                 */
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    std::string s = typeid(*this).name();
                    size_t n = s.find("cuvnet");
                    desc.label = desc.label + s.substr(n + 7);;
                }

                /**
                 * clean up temporary data.
                 *
                 * The default calls reset on the values of the parameters and
                 * on the deltas of the results.
                 */
                virtual void release_data(){
                    BOOST_FOREACH(Op::param_t& r, m_params){
                        r->value.reset();
                    }
                    BOOST_FOREACH(Op::result_t& r, m_results){
                        r->delta.reset();
                    }
                }

                private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        //ar & boost::serialization::base_object<boost::enable_shared_from_this<Op> >(*this);
                        ar & m_results & m_params;
                    }
        };
}
#endif /* __OP_HPP__ */
