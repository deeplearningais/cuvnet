#ifndef __ROW_SELECTOR_HPP__
#     define __ROW_SELECTOR_HPP__

#include <cuvnet/op.hpp>

namespace cuvnet
{
    /**
     * select a (random) row out of a matrix.
     *
     * Rows may be chosen randomly during fprop, if not supplied.
     * if multiple inputs are given, the same row is chosen for all of them.
     *
     * @ingroup Ops
     */
    class RowSelector
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                int  m_row; ///< the row to be selected
                bool m_random; ///< whether to choose row randomly in fprop
                bool m_copy; ///< if true, does not use view on inputs (might save some memory at cost of speed if input matrix is huge and not needed by other ops)

            public:
                RowSelector(){} ///< for serialization
                /**
                 * ctor.
                 * @param p0 first input
                 * @param row if -1, select row randomly in every fprop.
                 * @param copy if true, copy result before passing it to the next functor.
                 */
                RowSelector(result_t& p0, int row=-1, bool copy=false):Op(1,1),m_row(row),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                }
                /**
                 * ctor.
                 * @param p0 first input
                 * @param p1 second input
                 * @param row if -1, select row randomly in every fprop.
                 * @param copy if true, copy result before passing it to the next functor.
                 */
                RowSelector(result_t& p0, result_t& p1, int row=-1, bool copy=false):Op(2,2),m_row(row),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                    add_param(1,p1);
                }
                /**
                 * ctor.
                 * @param p0 first input
                 * @param p1 second input
                 * @param p2 third input
                 * @param row if -1, select row randomly in every fprop.
                 * @param copy if true, copy result before passing it to the next functor.
                 */
                RowSelector(result_t& p0, result_t& p1, result_t& p2, int row=-1, bool copy=false):Op(3,3),m_row(row),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                    add_param(1,p1);
                    add_param(2,p2);
                }
                /**
                 * ctor.
                 * @param p0 first input
                 * @param p1 second input
                 * @param p2 third input
                 * @param p3 fourth input
                 * @param row if -1, select row randomly in every fprop.
                 * @param copy if true, copy result before passing it to the next functor.
                 */
                RowSelector(result_t& p0, result_t& p1, result_t& p2, result_t& p3, int row=-1, bool copy=false):Op(3,3),m_row(row),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                    add_param(1,p1);
                    add_param(2,p2);
                    add_param(3,p3);
                }

                /**
                 * can be used to turn randomization off, eg for gradient testing.
                 */
                void set_random(bool b);

                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_row & m_random & m_copy;
                    }
        };

    /**
     * select a (random) batch of fixed size out of its inputs.
     *
     * One typical use would be to select random batches of the inputs \f$X\f$
     * and the target \f$Y\f$.
     *
     * @ingroup Ops
     */
    class RowRangeSelector
        : public Op{
            public:
                typedef Op::value_type    value_type;
                typedef Op::op_ptr        op_ptr;
                typedef Op::value_ptr     value_ptr;
                typedef Op::param_t       param_t;
                typedef Op::result_t      result_t;

            private:
                int  m_row; ///< the row to be selected
                unsigned int  m_nrows; ///< how many rows to choose
                bool m_random; ///< whether to choose row randomly in fprop
                bool m_copy; ///< if true, does not use view on inputs (might save some memory at cost of speed if input matrix is huge and not needed by other ops)

            public:
                RowRangeSelector(){} ///< for serialization

                /**
                 * ctor.
                 * @param p0 first input
                 * @param n_rows number of rows to select
                 * @param row if -1, select row randomly in every fprop.
                 * @param copy if true, copy result before passing it to the next functor.
                 */
                RowRangeSelector(result_t& p0, unsigned int n_rows, int row=-1, bool copy=false)
                    :Op(1,1),m_row(row),m_nrows(n_rows),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                }

                /**
                 * ctor.
                 * @param p0 first input
                 * @param p1 first input
                 * @param n_rows number of rows to select
                 * @param row if -1, select row randomly in every fprop.
                 * @param copy if true, copy result before passing it to the next functor.
                 */
                RowRangeSelector(result_t& p0, result_t& p1, unsigned int n_rows, int row=-1, bool copy=false)
                    :Op(2,2),m_row(row),m_nrows(n_rows),m_random(m_row<0),m_copy(copy){ 
                    add_param(0,p0);
                    add_param(1,p1);
                }

                /**
                 * can be used to turn randomization off, eg for gradient testing.
                 */
                void set_random(bool b);

                void fprop();
                void bprop();
                void _determine_shapes();
            private:
                friend class boost::serialization::access;
                template<class Archive>
                    void serialize(Archive& ar, const unsigned int version){
                        ar & boost::serialization::base_object<Op>(*this);
                        ar & m_row & m_nrows & m_random & m_copy;
                    }
        };

	
}

#endif /* __ROW_SELECTOR_HPP__ */
