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
                inline bool need_derivative()const{return m_need_derivative;}
                inline void need_derivative(bool b){m_need_derivative = b;}

                /**
                 * calculate recursively what needs to be calculated to
                 * derive this operator w.r.t. a set of parameters.
                 *
                 * The results are stored in the function itself.
                 *
                 * @param l the list of parameters w.r.t. which this op is to be derived
                 */
                bool set_calculate_derivative(const std::vector<Op*>&l);

                friend struct param_collector_visitor;
                friend struct toposort_visitor;
                friend struct determine_shapes_visitor;
                friend struct reset_value_set_flag;
                friend struct reset_delta_set_flag;
                friend struct define_graphviz_node_visitor;
                friend struct swiper;

                /**
                 * show all Ops to a (constant) visitor recursively
                 */
                template<class Visitor>
                    void visit(const Visitor& v){
                        if(!v.discover(this)) return;
                        v.preorder(this);
                        BOOST_FOREACH(Op::param_t& p, m_params){
                            BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                                r->get_op()->visit(v);
                            }
                        }
                        v.postorder(this);
                    }
                /**
                 * show all Ops to a  visitor recursively
                 */
                template<class Visitor>
                    void visit(Visitor& v){
                        if(!v.discover(this)) return;
                        v.preorder(this);
                        BOOST_FOREACH(Op::param_t& p, m_params){
                            BOOST_FOREACH(boost::shared_ptr<detail::op_result<value_type> > r, p->param_uses){
                                r->get_op()->visit(v);
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
                    desc.label = s.substr(n + 7);;
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
