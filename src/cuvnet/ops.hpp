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
#include <cuvnet/ops/log.hpp>
#include <cuvnet/ops/softmax.hpp>

namespace cuvnet
{
    inline
        Op::op_ptr operator*(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Multiply>(x->result(), y->result()); }
    inline
        Op::op_ptr operator+(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(), y->result(), 1.f,1.f); }
    inline
        Op::op_ptr operator+(Op::op_ptr x, float f)     { return boost::make_shared<AddScalar>(x->result(), f); }
    inline
        Op::op_ptr operator+(float f, Op::op_ptr x)     { return boost::make_shared<AddScalar>(f, x->result()); }
    inline
        Op::op_ptr operator-(float f, Op::op_ptr x)     { return boost::make_shared<SubtractFromScalar>(f, x->result()); }
    inline
        Op::op_ptr operator-(Op::op_ptr x)              { return boost::make_shared<Axpby>(x->result(), x->result(), 0.f,-1.f); }
    inline
        Op::op_ptr log(Op::op_ptr x)                    { return boost::make_shared<Log>(x->result()); }
    inline
        Op::op_ptr pow(Op::op_ptr x, float f)           { return boost::make_shared<Pow>(f, x->result()); }
    inline
        Op::op_ptr prod(Op::op_ptr x, Op::op_ptr y, char tx='n', char ty='n') { return boost::make_shared<Prod>(x->result(), y->result(), tx, ty); }
    inline
        Op::op_ptr tanh(Op::op_ptr x)                   { return boost::make_shared<Tanh>(x->result()); }
    inline
        Op::op_ptr logistic(Op::op_ptr x)               { return boost::make_shared<Logistic>(x->result()); }
    inline
        Op::op_ptr logistic(Op::op_ptr x, bool b)       { return boost::make_shared<Logistic>(x->result(),b); }
    inline
        Op::op_ptr axpby(float a, Op::op_ptr x, float b, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(),y->result(),a,b); }
    inline
        Op::op_ptr axpby(float a, Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(),y->result(),a,1.f); }
    inline
        Op::op_ptr axpby(Op::op_ptr x, float b, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(),y->result(),1.f,b); }
    inline
        Op::op_ptr add_rnd_normal(Op::op_ptr x, float f){ return boost::make_shared<Noiser>(x->result(),f, Noiser::NT_NORMAL); }
    inline
        Op::op_ptr zero_out(Op::op_ptr x, float f){ return boost::make_shared<Noiser>(x->result(),f, Noiser::NT_ZERO_OUT); }
    inline
        Op::op_ptr sum(Op::op_ptr x)                    { return boost::make_shared<Sum>(x->result()); }
    inline
        Op::op_ptr sum(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<SumMatToVec>(x->result(), ax==0?true:false ); }
    inline
        Op::op_ptr mean(Op::op_ptr x)                   { return boost::make_shared<Mean>(x->result()); }
    inline
        Op::op_ptr mat_plus_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax) { return boost::make_shared<MatPlusVec>(x->result(),v->result(), ax==0?false:true); }
}
#endif /* __OPS_HPP__ */