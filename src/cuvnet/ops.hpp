#ifndef __OPS_HPP__
#     define __OPS_HPP__

#include <cuvnet/ops/axpby.hpp>
#include <cuvnet/ops/identity.hpp>
#include <cuvnet/ops/input.hpp>
#include <cuvnet/ops/mat_plus_vec.hpp>
#include <cuvnet/ops/output.hpp>
#include <cuvnet/ops/pow.hpp>
#include <cuvnet/ops/prod.hpp>
#include <cuvnet/ops/tanh.hpp>
#include <cuvnet/ops/noiser.hpp>
#include <cuvnet/ops/sum.hpp>
#include <cuvnet/ops/multiply.hpp>
#include <cuvnet/ops/sum_mat_to_vec.hpp>
#include <cuvnet/ops/add_scalar.hpp>

namespace cuvnet
{
    inline
        Op::op_ptr operator*(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Multiply>(x->result(), y->result()); }
    inline
        Op::op_ptr operator+(Op::op_ptr x, float f)     { return boost::make_shared<AddScalar>(x->result(), f); }
    inline
        Op::op_ptr operator+(float f, Op::op_ptr x)     { return boost::make_shared<AddScalar>(f, x->result()); }
    inline
        Op::op_ptr operator-(float f, Op::op_ptr x)     { return boost::make_shared<SubtractFromScalar>(f, x->result()); }
    inline
        Op::op_ptr pow(Op::op_ptr x, float f)           { return boost::make_shared<Pow>(f, x->result()); }
    inline
        Op::op_ptr prod(Op::op_ptr x, Op::op_ptr y, char tx='n', char ty='n') { return boost::make_shared<Prod>(x->result(), y->result(), tx, ty); }
    inline
        Op::op_ptr tanh(Op::op_ptr x)                   { return boost::make_shared<Tanh>(x->result()); }
    inline
        Op::op_ptr axpby(float a, Op::op_ptr x, float b, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(),y->result(),a,b); }
    inline
        Op::op_ptr add_rnd_normal(Op::op_ptr x, float f){ return boost::make_shared<Noiser>(x->result(),f); }
    inline
        Op::op_ptr sum(Op::op_ptr x)                    { return boost::make_shared<Sum>(x->result()); }
}
#endif /* __OPS_HPP__ */
