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
                typedef matrix value_type;
                typedef cow_ptr<value_type>                 value_ptr;
                typedef boost::shared_ptr<Op>               op_ptr;
                typedef boost::weak_ptr<detail::op_param<value_type> >     weak_param_t;
                typedef boost::shared_ptr<detail::op_param<value_type> >        param_t;
                typedef boost::shared_ptr<detail::op_result<value_type> >      result_t;
            protected:
                std::vector<param_t>  m_params;
                std::vector<result_t> m_results;
                bool                  m_need_derivative;
                bool                  m_need_result;

            public:
                /**
                 * Default constructor (you shouldn't use this, only implicitly during deserialization!)
                 */
                Op();
                Op(unsigned int n_params, unsigned int n_results);
                virtual ~Op();

                Op& detach_from_params();
                Op& detach_from_results();

                /** 
                 * returns a reference to the i-th result
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
                void set_n_params(unsigned int n);
                void set_n_results(unsigned int n);
                inline unsigned int get_n_params(){ return m_params.size(); }
                inline unsigned int get_n_results(){ return m_results.size(); }
                void add_param(unsigned int idx, result_t& p);
                virtual bool need_derivative()const{return m_need_derivative;}
                inline void need_derivative(bool b){m_need_derivative = b;}
                virtual bool need_result()const{return m_need_result;}
                inline void need_result(bool b){m_need_result = b;}

                /**
                 * calculate recursively what needs to be calculated to
                 * derive this operator w.r.t. a set of parameters.
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
                 * show all Ops to a (constant) visitor recursively
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
                                    r.lock()->get_op()->visit(v, false);
                                }
                            }
                        }
                        v.postorder(this);
                    }
                /**
                 * show all Ops to a  visitor recursively
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
                 * user-supplied function: calculate results of this op
                 */
                virtual void fprop()=0;
                /**
                 * user-supplied function: backpropagate results of this op
                 */
                virtual void bprop()=0;

                /**
                 * virtual function: determine the shape for each result.
                 *
                 * The default works for ops with only one input:
                 * the shape of the input is simply passed to each result.
                 */
                virtual void _determine_shapes();

                /**
                 * modify the graphviz node description string
                 */
                virtual void _graphviz_node_desc(detail::graphviz_node& desc)const{
                    std::string s = typeid(*this).name();
                    size_t n = s.find("cuvnet");
                    desc.label = desc.label + s.substr(n + 7);;
                }

                /**
                 * clean up temporary data
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
