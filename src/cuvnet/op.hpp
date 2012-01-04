#ifndef __OP_HPP__
#     define __OP_HPP__
#include <list>
#include <map>
#include <boost/weak_ptr.hpp>
#include <cuvnet/common.hpp>
#include <cuvnet/smart_ptr.hpp>

namespace cuvnet
{
	typedef char param_name_t;
	class Op;


	namespace detail{
		/**
		 * infos about an argument of an operator
		 */
		template<class T>
		struct op_arg{
			boost::shared_ptr<Op> op;    /// the operator whose result we are using
			param_name_t        param;   /// the name of the result we're using
			ce_ptr<T>           value;   /// contains the value of the parameter
			bool                derive;  /// whether we need to derive w.r.t. this argument
			op_arg(param_name_t n, boost::shared_ptr<Op> o):op(o),param(n),derive(false){}
			op_arg():derive(false){}
			bool operator==(const op_arg& o)const{ return op==o.op && param==o.param; }
		};
		/**
		 * infos about the op that uses the result of an operator
		 */
		struct op_result_use{
			boost::weak_ptr<Op> op;    /// the operator whose result we are using
			param_name_t        param; /// the name of the result we're using
			op_result_use(param_name_t n, boost::shared_ptr<Op> o):op(o),param(n){}
			op_result_use():param(0){}
			bool operator==(const op_result_use& o)const{ return boost::shared_ptr<Op>(op)==boost::shared_ptr<Op>(o.op) && param==o.param; }
		};
		/**
		 * a result of an operator
		 */
		template<class T>
		struct op_result{
			ce_ptr<T>        result; /// the result value
			ce_ptr<T>        delta;  /// the delta from consecutive ops
			std::vector<op_result_use> dst_ops; /// all users
			/**
			 * add a user of this result
			 */
			inline void add_user(param_name_t n, boost::shared_ptr<Op> o){
				dst_ops.push_back(op_result_use(n,o));
			}
			/**
			 * return the number of users of this result
			 */
			inline unsigned int cnt_users(){ return dst_ops.size(); }
			
			/**
			 * construct `ce_ptr's using default constructor
			 */
			op_result()
			: result(new T())
			, delta (new T()){
			}
		};
	}


	/**
	 * an abstract functor which gets \f$n\f$ parameters and generates
	 * \f$m\f$ results.
	 */
	class Op
	: public boost::enable_shared_from_this<Op>
	{
		public:
			typedef matrix value_type;
		private:
			friend struct toposorter;
			friend struct swiper;
			std::vector<detail::op_result<value_type> > m_results;
			std::vector<detail::op_arg<value_type> >                 m_params;

			/**
			 * push results to users
			 */
			inline void push_results(){
				typedef std::vector<detail::op_result_use> arg_vec;
				for(std::vector<detail::op_result<value_type> >::iterator it=m_results.begin(); it!=m_results.end(); ++it){
					for(arg_vec::iterator ait=it->dst_ops.begin(); ait!=it->dst_ops.end(); ++ait){
						boost::shared_ptr<Op> ptr(ait->op);
						ptr->m_params[ait->param].value = it->result;
					}
				}
			}

			/**
			 * recursively remove all result-uses.
			 *
			 * result-uses can be rebuilt using arborize_build
			 */
			inline void arborize_clear(){
				for(std::vector<detail::op_result<value_type> >::iterator it=m_results.begin(); it!=m_results.end(); ++it){
					it->dst_ops.clear();
				}
				for(std::vector<detail::op_arg<value_type> >::iterator it=m_params.begin(); it!=m_params.end(); ++it){
					boost::shared_ptr<Op> itop(it->op);
					itop->arborize_clear();
				}
			}
			/**
			 * recursively notify ops which result are used by whom
			 *
			 * assumes that arborize_clear has been called before
			 */
			inline void arborize_build(){
				unsigned int cnt=0;
				for(std::vector<detail::op_arg<value_type> >::iterator it=m_params.begin(); it!=m_params.end(); ++it,++cnt){
					boost::shared_ptr<Op> itop(it->op);
					itop->add_result_user(it->param, shared_from_this(), cnt);
					itop->arborize_build();
				}
			}
		protected:
			Op(unsigned int n_params, unsigned int n_results){
				set_n_params(n_params);
				set_n_results(n_results);
			}
			inline Op& set_n_params(unsigned int n){ m_params.resize(n); return *this;}
			inline Op& set_n_results(unsigned int n){ m_results.resize(n);return *this;}

		public:
			Op& arborize(){
				arborize_clear();
				arborize_build();
				return *this;
			}
			inline unsigned int get_n_params(unsigned int n)const{ return m_params.size();}
			inline unsigned int get_n_results(unsigned int n)const{ return m_results.size();}

			inline Op& add_result_user(const param_name_t& n, const boost::shared_ptr<Op>& op, const param_name_t& op_n){
				m_results[n].dst_ops.push_back(detail::op_result_use(op_n,op));
				return *this;
			}
			inline Op& set_param(const param_name_t& n, const boost::shared_ptr<Op>& op, const param_name_t& op_n){
				m_params[n] = detail::op_arg<value_type> (op_n, op);
				return *this;
			}
			inline bool want_partial(const param_name_t& n=0){
				return m_params[n].derive;
			}
			inline ce_ptr<value_type>& get_result_value(const param_name_t& n=0){
				detail::op_result<value_type>& ores = m_results[n];
				return ores.result;
			}
			inline ce_ptr<value_type>& get_delta_value(const param_name_t& n=0){
				detail::op_result<value_type>& ores = m_results[n];
				return ores.delta;
			}
			inline ce_ptr<value_type>& get_param_value(const param_name_t& n=0){
				detail::op_arg<value_type> & oa = m_params[n];
				return oa.value;
			}
			inline boost::shared_ptr<Op> get_param_op(const param_name_t& n=0){
				return m_params[n].op;
			}
			inline ce_ptr<value_type>& get_param_delta(const param_name_t& n=0){
				const detail::op_arg<value_type> & oa = m_params[n];
				return oa.op->get_delta_value(oa.param);
			}
			/**
			 * call when op does not use the n-th result of this as its op_n-th input anymore 
			 */
			inline Op& remove_result_user(const param_name_t& n, const boost::shared_ptr<Op>& op, const param_name_t& op_n){
				std::vector<detail::op_result_use>& dst_ops = m_results[n].dst_ops;
				dst_ops.erase(  std::remove_if(
							dst_ops.begin(), dst_ops.end(), 
							std::bind2nd(std::equal_to<detail::op_result_use>(), detail::op_result_use(op_n,op))), 
						dst_ops.end());
			}
			virtual void fprop()=0;
			virtual void bprop()=0;

			/**
			 * return all the parameters of this function (e.g., the filters of a convolution op)
			 */
			virtual void my_params(std::list<boost::shared_ptr<const Op> >&)const{};

			/**
			 * determines a list of all parameters of an operator (recursively)
			 */
			inline void params(std::list<boost::shared_ptr<const Op> >&l)const{
				my_params(l);
				for(std::vector<detail::op_arg<value_type> >::const_iterator it=m_params.begin(); it!=m_params.end(); ++it)
					boost::shared_ptr<Op>(it->op)->params(l);
			};
			/**
			 * calculate recursively what needs to be calculated to
			 * derive this operator w.r.t. a set of parameters.
			 *
			 * this function is not particularly efficient, but we
			 * assume it is not called very often. The results are
			 * stored in the function itself.
			 *
			 * @param l the list of parameters w.r.t. which this op is to be derived
			 */
			inline void derivative(const std::list<boost::shared_ptr<const Op> >&l){
				for(std::vector<detail::op_arg<value_type> >::iterator it=m_params.begin(); it!=m_params.end(); ++it){
					std::list<boost::shared_ptr<const Op> > it_params;
					boost::shared_ptr<Op> itop(it->op);

					itop->params(it_params);
					it->derive = it_params.end() 
						!= std::find_first_of(
								it_params.begin(), it_params.end(),
								l.begin(), l.end());
					itop->derivative(l);
				}
			}
			template<class Visitor>
			bool visit_preorder(Visitor& v){
				if(!v(shared_from_this())) 
					return false;
				for(std::vector<detail::op_arg<value_type> >::iterator it=m_params.begin(); it!=m_params.end(); ++it){
					if(!it->op->visit_preorder(v))
						return false;
				}
				return true;
			}

			template<class T>
			bool isa()const{ return NULL != dynamic_cast<const T*>(this); }
	};
	class Input
	: public Op{
		public:
			Input():Op(0,1){}
			void fprop(){}
			void bprop(){}
			void my_params(std::list<boost::shared_ptr<const Op> >& l)const{ l.push_back(shared_from_this());};
	};
	class EltwiseLossFunction
	: public Op
	{
		public:
			EltwiseLossFunction():Op(1,1){}
		private:
	};

	/**
	 * function:   \f$ f(\mathbf(x)) = \sum_i 2x_i^2\f$
	 *
	 * derivative: \f$ \partial f(\mathbf{x}/\partial x_i = x_i
	 */
	class SquaredLossFunction
	: public EltwiseLossFunction
	{
		public:
			SquaredLossFunction(const boost::shared_ptr<Op>& p0, const param_name_t& p0_n=0){
				this->set_param(0,p0,p0_n);
			}

			void fprop(){
				// determine correct shape of output
				get_result_value(0).data(this).resize(cuv::extents[1]); 
				float n = cuv::norm2(get_param_value(0).cdata());
				get_result_value(0).data(this)[0] = 4*n*n;

				// do not unlock input, we need it again in bprop
			}
			void bprop(){
				boost::shared_ptr<Op> dop = get_param_op(0);
				get_param_delta(0)        = get_param_value(0);

				get_param_value(0).unlock(this);    // we do not need input anymore
				get_param_delta(0).lock(dop.get()); // p0 needs delta
			}
	};
	/**
	 * function \f$f_i(\mathbf{x}, \mathbf{y}) = x_i-y_i \f$
	 *
	 * derivative \f$\partial f_i(\mathbf{x}, \mathbf{y})/\partial x_i =  1 \f$
	 *
	 * derivative \f$\partial f_i(\mathbf{x}, \mathbf{y})/\partial y_i = -1 \f$
	 */
	class DifferenceOp
	: public Op
	{
		public:
			DifferenceOp(const boost::shared_ptr<Op>& p0, const boost::shared_ptr<Op>& p1, const param_name_t& p0_n=0, const param_name_t& p1_n=0)
			: Op(2,1)
			{
				this->set_param(0,p0,p0_n);

				this->set_param(1,p1,p1_n);
			}
		public:
			void fprop(){
				assert(get_param_value(0).cdata().shape()==get_param_value(1).cdata().shape());

				get_result_value(0).data(this).resize(get_param_value(0)->shape()); 

				get_result_value(0) = get_param_value(0);
				matrix& r           = get_result_value(0).data(this);
				const matrix& p1    = get_param_value(1).cdata();
				cuv::apply_binary_functor(r,p1,cuv::BF_SUBTRACT);

				get_param_value (0).unlock(this);
				get_param_value (1).unlock(this);
				get_result_value(0).unlock(this);
			}
			void bprop(){
				boost::shared_ptr<Op> dop0 = get_param_op(0);
				boost::shared_ptr<Op> dop1 = get_param_op(1);


				if(want_partial(0)){
					if(!get_param_delta(0).flagged()) // we're the first to set a value
						get_param_delta(0)                   = get_delta_value(0); // don't copy
					else                              // some value is already set here
						get_param_delta(0).data(dop0.get()) += get_delta_value(0).cdata(); 

					get_param_delta(0).flag();        // notify that we have set a value
					get_param_delta(0).lock(dop0.get()); // p0 needs to read its delta!
				}
				if(want_partial(1)){
					if(!get_param_delta(1).flagged()) // we're the first to set a value
					{
						get_param_delta(1).data(dop1.get()) -= get_delta_value().cdata();
					}
					else{
						get_param_delta(1) = get_delta_value();       // don't copy 
						get_param_delta(1).data(dop1.get()) *= -1.f;  // copies if necessary
					}
					get_param_delta(1).flag();
					get_param_delta(1).lock(dop1.get()); // p1 needs to read its delta!
				}
				get_delta_value(0).unlock(this); // never need this again after this function
			}
	};

	/**
	 * function \f$ f_i(x) = tanh(x_i)\f$
	 *
	 * derivative \f$ d f_i(x)/d x_i = 1 - (x_i)^2\f$
	 */
	class TanhOp
	: public Op{
		public:
			TanhOp(boost::shared_ptr<Op> p0, const param_name_t& p0_n=0):Op(1,1){ 
				this->set_param(0,p0,p0_n);
			}
			void fprop(){
				// determine correct shape of output
				get_result_value(0) = get_param_value(0); 

				cuv::apply_scalar_functor(
						get_result_value(0).data(this),
						get_param_value(0).cdata(),
					       	cuv::SF_TANH);

				if(!want_partial(0)) // unlock result if we don't need it for bprop!
					get_result_value(0).unlock(this);
				get_param_value(0).unlock(this); // never need this anyway!
			}
			void bprop(){
				if(!want_partial(0))
					// we don't even need to unlock get_delta_value(0), since it should not have been calculated in the 1st place!
					return; 

				get_param_delta(0)        = get_result_value(0);
				boost::shared_ptr<Op> dop = get_param_op(0);
				value_type&       d0      = get_param_delta(0).data(dop.get()); 

				if(! get_param_delta(0).flagged()){
					const value_type& r   = get_result_value(0).cdata();
					d0.resize(r.shape()); 
					cuv::apply_scalar_functor(d0, r, cuv::SF_DTANH);  // change d0 directly
				}else{
					value_type& r         = get_result_value(0).data(this);       // change i, then add to d
					cuv::apply_scalar_functor(r, cuv::SF_DTANH);
					d0+= r;
				}

				get_param_delta(0).lock(dop.get()); // p0 wants to read that, we can assume
				get_result_value(0).unlock(this);    // don't need input anymore 
			}
	};
	class PowOp
	: public Op{
		public:
		private:
			float m_exponent;
		public:
			PowOp(float exponent, boost::shared_ptr<Op> p0, const param_name_t& p0_n=0):Op(1,1),m_exponent(exponent){ 
				assert(exponent!=1.f); 
				this->set_param(0,p0,p0_n);
			}
			void fprop(){
				// assume that input has been locked for us:
				// - do NOT unlock input (for bprop)
				// - do NOT change input (for bprop)
				
				// determine correct shape of output
				get_result_value(0).data(this).resize(get_param_value(0)->shape()); 

				cuv::apply_scalar_functor(
						get_result_value(0).data(this),
						get_param_value(0).cdata(),
					       	cuv::SF_POW, m_exponent);

				if(!want_partial(0)){
					// unlock input if we don't need it for bprop!
					get_param_value(0).unlock(this);
				}
			}
			void bprop(){
				if(!want_partial(0))
					// we don't even need to unlock get_delta_value(0), since it should not have been calculated in the 1st place!
					return; 

				get_param_delta(0)        = get_param_value(0);
				boost::shared_ptr<Op> dop = get_param_op(0);
				value_type&       d       = get_param_delta(0).data(dop.get()); 

				if(! get_param_delta(0).flagged()){
					const value_type& i   = get_param_value(0).cdata();
					d.resize(i.shape()); 
					cuv::apply_scalar_functor(d, i, cuv::SF_DPOW, m_exponent);  // change d directly
				}else{
					value_type& i         = get_param_value(0).data(this);      // change i, then add to d
					cuv::apply_scalar_functor(i, cuv::SF_DPOW, m_exponent);
					d+= i;
				}

				get_param_delta(0).lock(dop.get()); // p0 wants to read that, we can assume
				get_param_value(0).unlock(this);    // don't need input anymore 
			}
	};

	/**
	 * function \f$ f(X,Y) = XY\f$
	 *
	 */
	class ProdOp
	: public Op
	{
		private:
			char m_t0, m_t1;
		public:
			ProdOp(const boost::shared_ptr<Op>& p0, const boost::shared_ptr<Op>& p1, const char t0, const char t1, const param_name_t& p0_n=0, const param_name_t& p1_n=0)
			: Op(2,1),m_t0(t0),m_t1(t1)
			{
				this->set_param(0,p0,p0_n);

				this->set_param(1,p1,p1_n);
			}
		public:
			void fprop(){
				const value_type& p0val = get_param_value(0).cdata();
				const value_type& p1val = get_param_value(1).cdata();
				unsigned int n = m_t0=='n' ? p0val.shape(0) : p0val.shape(1);
				unsigned int m = m_t1=='n' ? p1val.shape(0) : p1val.shape(1);

				value_type& res = get_result_value(0).data(this).resize(cuv::extents[n][m]); 
				cuv::prod(res,p0val,p1val,m_t0,m_t1);

				// dot(W,X)
				if(!want_partial(0))                        // d/dW = dot(X,delta)
					get_param_value (1).unlock(this);   //   --> forget X
				if(!want_partial(1))                        // d/dX = dot(W,delta)
					get_param_value (0).unlock(this);   //   --> forget W
				get_result_value(0).unlock(this);
			}
			void bprop(){
				boost::shared_ptr<Op> dop0 = get_param_op(0);
				boost::shared_ptr<Op> dop1 = get_param_op(1);

				const value_type& delta = get_delta_value().cdata();
				unsigned int n = delta.shape(0);
				unsigned int m = delta.shape(1);

				if(want_partial(0)){
					float oldfact = get_param_delta(0).flagged() ? 1.f : 0.f;

					prod(           get_param_delta(0).data(dop0.get()),
							delta,
							get_param_value(1).cdata(),
							'n',m_t1=='n'?'t':'n',1.f,oldfact); // oldfact times whatever is in get_param_delta(0) already

					get_param_delta(0).flag();        // notify that we have set a value
					get_param_delta(0).lock(dop0.get()); // p0 needs to read its delta!
				}
				if(want_partial(1)){
					float oldfact = get_param_delta(1).flagged() ? 1.f : 0.f;

					prod(           get_param_delta(1).data(dop1.get()),
							get_param_value(0).cdata(),
							delta,
							't',m_t0=='n'?'n':'t',1.f,oldfact); // oldfact times whatever is in get_param_delta(0) already
					get_param_delta(1).flag();
					get_param_delta(1).lock(dop1.get()); // p1 needs to read its delta!
				}
				get_delta_value(0).unlock(this); // never need this again after this function
			}
	};

	/**
	 *
	 * function \f$ f_{ij}(X, \vec b) = X_{ij}+b_j\f$ (adds a column)
	 *
	 * if t1 is 't', then we have instead
	 *
	 * function \f$ f_{ij}(X, \vec b) = X_{ij}+b_i\f$ (adds a row)
	 *
	 */
	class MatrixPlusVecOp
	: public Op
	{
		private:
			char m_t1;
		public:
			MatrixPlusVecOp(const boost::shared_ptr<Op>& p0, const boost::shared_ptr<Op>& p1, const char t1, const param_name_t& p0_n=0, const param_name_t& p1_n=0)
			: Op(2,1),
			  m_t1(t1)
			{
				this->set_param(0,p0,p0_n);
				this->set_param(1,p1,p1_n);
			}
		public:
			void fprop(){
				assert( get_param_value(0).cdata().shape(0) ==
					get_param_value(1).cdata().shape(m_t1=='n'?0:1));

				get_result_value(0) = get_param_value(0);
				if(m_t1=='n')
					cuv::matrix_plus_col(get_result_value(0).data(this),get_param_value(1).cdata());
				else
					cuv::matrix_plus_row(get_result_value(0).data(this),get_param_value(1).cdata());

				get_param_value (0).unlock(this);
				get_param_value (1).unlock(this);
				get_result_value(0).unlock(this);
			}
			void bprop(){
				boost::shared_ptr<Op> dop0 = get_param_op(0);
				boost::shared_ptr<Op> dop1 = get_param_op(1);

				if(want_partial(0)){
					if(!get_param_delta(0).flagged()) // we're the first to set a value
						get_param_delta(0)                   = get_delta_value(0); // don't copy
					else                              // some value is already set here
						get_param_delta(0).data(dop0.get()) += get_delta_value(0).cdata(); 

					get_param_delta(0).flag();        // notify that we have set a value
					get_param_delta(0).lock(dop0.get()); // p0 needs to read its delta!
				}
				if(want_partial(1)){
					float oldfact           = get_param_delta(1).flagged() ? 1 : 0;
					const value_type& delta = get_delta_value(1).cdata();
					value_type& d1          = get_param_delta(1).data(dop1.get());

					if(m_t1=='n'){
						d1.resize(cuv::extents[delta.shape(0)]);
						cuv::reduce_to_col(d1, delta, cuv::RF_ADD, 1.f, oldfact);
					}else{
						d1.resize(cuv::extents[delta.shape(1)]);
						cuv::reduce_to_row(d1, delta, cuv::RF_ADD, 1.f, oldfact);
					}
					get_param_delta(1).flag();
					get_param_delta(1).lock(dop1.get()); // p1 needs to read its delta!
				}
				get_delta_value(0).unlock(this); // never need this again after this function
			}
	};

	boost::shared_ptr<Op>
	make_mlp_op(unsigned int layersize, bool want_bias, cuv::ScalarFunctor sf, const boost::shared_ptr<Op>& p0, const param_name_t& p0_n=0){
		// TODO: how to infer shape??
		boost::shared_ptr<Input>  weights(new Input());
		boost::shared_ptr<Op>     func(new ProdOp(p0,weights,p0_n,0));
		if(want_bias){
			boost::shared_ptr<Input>  bias(new Input());
			func.reset(new MatrixPlusVecOp(func,bias, 't')); // we assume that examples are stored in rows, so we need to 't' the bias
		}
		switch(sf){
			case cuv::SF_TANH: func.reset(new TanhOp(func)); break;
			default: assert(false && "Unknown scalar functor!");
		}
		return func;
	}

	struct toposorter{
		typedef std::vector<boost::shared_ptr<Op> > sorted_seq_t;
		sorted_seq_t sorted;
		std::map<void*, bool>             marked;
		bool operator()(const boost::shared_ptr<Op>& o){
			if(marked.find(o.get())==marked.end()){
				marked[o.get()] = true;
				typedef std::vector<detail::op_result<Op::value_type> > res_vec;
				res_vec& results = o->m_results;
				typedef std::vector<detail::op_result_use> arg_vec;
				for(res_vec::iterator resit = results.begin(); resit!=results.end();++resit){
					for(arg_vec::iterator ait=resit->dst_ops.begin(); ait!=resit->dst_ops.end(); ++ait){
						this->operator()(boost::shared_ptr<Op>(ait->op));
					}
				}
				sorted.push_back(o);
			}
			return true;
		}
	};

	struct swiper{
		private:
			toposorter            m_topo;
			boost::shared_ptr<Op> m_func;
		public:
			swiper(boost::shared_ptr<Op> op):m_func(op){
				m_func->visit_preorder(m_topo);
			}
			void fprop(){
				for( toposorter::sorted_seq_t::iterator it = m_topo.sorted.begin();
						it != m_topo.sorted.end();
						++it){
					// calculate all results of this op
					(*it)->fprop();
					
					// The results are protected by a copy-elimination pointer (ce_ptr)
					// If there are multiple users of a result, 
					// we need to lock the result for all potential users.

					// loop over all results
					for(std::vector<detail::op_result<Op::value_type> >::iterator rit = (*it)->m_results.begin();
							rit != (*it)->m_results.end();
							++rit){
						// loop over all users of this result
						for(std::vector<detail::op_result_use>::iterator uit = rit->dst_ops.begin();
								uit != rit->dst_ops.end();
								++uit){
							//(*rit)->_internal_accept_owner
						}
					}
				}
			}
			void bprop(){
				// same thing reversed
				for( toposorter::sorted_seq_t::reverse_iterator it = m_topo.sorted.rbegin();
						it != m_topo.sorted.rend();
						++it){
					(*it)->bprop();
				}
			}
	};
}
#endif /* __OP_HPP__ */
