#ifndef __OPS_HPP__
#     define __OPS_HPP__

#include <cmath>
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
#include <cuvnet/ops/atan2.hpp>
#include <cuvnet/ops/sum_mat_to_vec.hpp>
#include <cuvnet/ops/add_scalar.hpp>
#include <cuvnet/ops/mult_scalar.hpp>
#include <cuvnet/ops/log.hpp>
#include <cuvnet/ops/exp.hpp>
#include <cuvnet/ops/abs.hpp>
#include <cuvnet/ops/softmax.hpp>
#include <cuvnet/ops/convolve.hpp>
#include <cuvnet/ops/reshape.hpp>
#include <cuvnet/ops/row_selector.hpp>
#include <cuvnet/ops/classification_error.hpp>

namespace cuvnet
{
    inline
        Op::op_ptr operator*(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Multiply>(x->result(), y->result()); }
    inline
        Op::op_ptr operator+(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(), y->result(), 1.f,1.f); }
    inline
        Op::op_ptr operator-(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(), y->result(), 1.f,-1.f); }
    inline
        Op::op_ptr operator+(Op::op_ptr x, float f)     { return boost::make_shared<AddScalar>(x->result(), f); }
    inline
        Op::op_ptr operator+(float f, Op::op_ptr x)     { return boost::make_shared<AddScalar>(f, x->result()); }
    inline
        Op::op_ptr operator*(Op::op_ptr x, float f)     { return boost::make_shared<MultScalar>(x->result(), f); }
    inline
        Op::op_ptr operator*(float f, Op::op_ptr x)     { return boost::make_shared<MultScalar>(f, x->result()); }
    inline
        Op::op_ptr operator/(Op::op_ptr x, float f)     { return boost::make_shared<MultScalar>(x->result(), 1.f/f); }
    inline
        Op::op_ptr operator-(float f, Op::op_ptr x)     { return boost::make_shared<SubtractFromScalar>(f, x->result()); }
    inline
        Op::op_ptr operator-(Op::op_ptr x)              { return boost::make_shared<Axpby>(x->result(), x->result(), 0.f,-1.f); }
    inline
        Op::op_ptr atan2(Op::op_ptr y, Op::op_ptr x)    { return boost::make_shared<Atan2>(y->result(), x->result()); }
    inline
        Op::op_ptr log(Op::op_ptr x)                    { return boost::make_shared<Log>(x->result()); }
    inline
        Op::op_ptr square(Op::op_ptr x)                 { return boost::make_shared<Pow>(2.f, x->result()); }
    inline
        Op::op_ptr sqrt(Op::op_ptr x)                   { return boost::make_shared<Pow>(.5f, x->result()); }
    inline
        Op::op_ptr pow(Op::op_ptr x, float f)           { return boost::make_shared<Pow>(f, x->result()); }
    inline
        Op::op_ptr abs(Op::op_ptr x, float eps=0.0001f) { return boost::make_shared<Abs>(x->result(), eps); }
    inline
        Op::op_ptr exp(float f, Op::op_ptr x)           { return boost::make_shared<Exp>(f, x->result()); }
    inline
        Op::op_ptr prod(Op::op_ptr x, Op::op_ptr y, char tx='n', char ty='n') { return boost::make_shared<Prod>(x->result(), y->result(), tx, ty); }
    inline
        boost::shared_ptr<Sink> sink(Op::op_ptr x, unsigned int res=0){ return boost::make_shared<Sink>(x->result(res)); }
    inline
        boost::shared_ptr<Sink> sink(const std::string& name, Op::op_ptr x, unsigned int res=0){ return boost::make_shared<Sink>(name, x->result(res)); }
    inline
        Op::op_ptr sin(Op::op_ptr x)                    { return boost::make_shared<Sin>(x->result()); }
    inline
        Op::op_ptr cos(Op::op_ptr x)                    { return boost::make_shared<Cos>(x->result()); }
    inline
        Op::op_ptr tanh(Op::op_ptr x)                   { return boost::make_shared<Tanh>(x->result()); }
    inline
        Op::op_ptr logistic(Op::op_ptr x)               { return boost::make_shared<Logistic>(x->result()); }
    inline
        Op::op_ptr logistic(Op::op_ptr x, bool b)       { return boost::make_shared<Logistic>(x->result(),b); }
    inline
        Op::op_ptr neg_log_cross_entropy_of_logistic(Op::op_ptr x, Op::op_ptr y) { return boost::make_shared<NegCrossEntropyOfLogistic>(x->result(), y->result()); }
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
        Op::op_ptr sum_to_vec(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<SumMatToVec>(x->result(), ax ); }
    inline
        Op::op_ptr mean(Op::op_ptr x)                   { return boost::make_shared<Mean>(x->result()); }
    inline
        Op::op_ptr mat_plus_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax) { return boost::make_shared<MatPlusVec>(x->result(),v->result(), ax); }
    inline
        Op::op_ptr mat_times_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax) { return boost::make_shared<MatTimesVec>(x->result(),v->result(), ax); }
    inline
        Op::op_ptr convolve(Op::op_ptr img, Op::op_ptr flt, bool padding=false) { return boost::make_shared<Convolve>(img->result(),flt->result(), padding); }
    inline
        Op::op_ptr reorder_for_conv(Op::op_ptr img) { return boost::make_shared<ReorderForConv>(img->result()); }
    inline
        Op::op_ptr reorder_from_conv(Op::op_ptr img) { return boost::make_shared<ReorderFromConv>(img->result()); }
    inline
        Op::op_ptr flatten(Op::op_ptr img, unsigned int outdim=1) { return boost::make_shared<Flatten>(img->result(),outdim); }
    inline
        Op::op_ptr local_pool(Op::op_ptr img, cuv::alex_conv::pool_type pt) { return boost::make_shared<LocalPooling>(img->result(),pt); }

    template<std::size_t D>
    inline
        Op::op_ptr reshape(Op::op_ptr img, const cuv::extent_gen<D>& eg) { return boost::make_shared<Reshape>(img->result(),eg); }

    inline
        Op::op_ptr softmax(Op::op_ptr img, unsigned int dim=0){ return boost::make_shared<Softmax>(img->result(), dim); }
    inline
        Op::op_ptr multinomial_logistic_loss(Op::op_ptr x, Op::op_ptr target, unsigned int dim=0){ return boost::make_shared<MultinomialLogisticLoss>(x->result(), target->result(), dim); }
    inline
        Op::op_ptr row_select(Op::op_ptr p0, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), row); }
    inline
        Op::op_ptr row_select(Op::op_ptr p0, Op::op_ptr p1, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), p1->result(), row); }
    inline
        Op::op_ptr row_select(Op::op_ptr p0, Op::op_ptr p1, Op::op_ptr p2, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), p1->result(), p2->result(), row); }
    inline
        Op::op_ptr row_select(Op::op_ptr p0, Op::op_ptr p1, Op::op_ptr p2, Op::op_ptr p3, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), p1->result(), p2->result(), p3->result(), row); }
    inline 
        Op::op_ptr result(Op::op_ptr p0, unsigned int result=0){ return boost::make_shared<Pipe>(p0->result(result), result); }

    inline
        Op::op_ptr classification_loss(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<ClassificationLoss>(x->result(),y->result()); }
    inline
        Op::op_ptr classification_loss(boost::shared_ptr<Sink> x, Op::op_ptr y){ return boost::make_shared<ClassificationLoss>(x->result(),y->result()); }

    inline 
        Op::op_ptr add_to_param(Op::op_ptr dst, Op::op_ptr src, int param=0, int result=0){
            dst->param(param)->param_uses.push_back(src->result(result));
            src->result(result)->result_uses.push_back(dst->param(param));
            return dst;
        }
}
#endif /* __OPS_HPP__ */
