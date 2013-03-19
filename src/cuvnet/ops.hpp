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
#include <cuvnet/ops/transpose.hpp>
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
#include <cuvnet/ops/rectified_linear.hpp>
#include <cuvnet/ops/classification_error.hpp>
#include <cuvnet/ops/debug.hpp>
#include <cuvnet/ops/subtensor.hpp>
#include <cuvnet/ops/concatenate.hpp>

namespace cuvnet
{
    /// @addtogroup convenience_funcs
    /// Convenience functions for combining Ops to create more complex Ops.
    /// @{
    
    /// construct a Multiply object
    inline
        Op::op_ptr operator*(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Multiply>(x->result(), y->result()); }
    /// construct a Axpby object
    inline
        Op::op_ptr operator+(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(), y->result(), 1.f,1.f); }
    /// construct a Axpby object
    inline
        Op::op_ptr operator-(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(), y->result(), 1.f,-1.f); }
    /// construct a AddScalar object
    inline
        Op::op_ptr operator+(Op::op_ptr x, float f)     { return boost::make_shared<AddScalar>(x->result(), f); }
    /// construct a AddScalar object
    inline
        Op::op_ptr operator+(float f, Op::op_ptr x)     { return boost::make_shared<AddScalar>(f, x->result()); }
    /// construct a MultScalar object
    inline
        Op::op_ptr operator*(Op::op_ptr x, float f)     { return boost::make_shared<MultScalar>(x->result(), f); }
    /// construct a MultScalar object
    inline
        Op::op_ptr operator*(float f, Op::op_ptr x)     { return boost::make_shared<MultScalar>(f, x->result()); }
    /// construct a MultScalar object
    inline
        Op::op_ptr operator/(Op::op_ptr x, float f)     { return boost::make_shared<MultScalar>(x->result(), 1.f/f); }
    /// construct a SubtractFormScalar object
    inline
        Op::op_ptr operator-(float f, Op::op_ptr x)     { return boost::make_shared<SubtractFromScalar>(f, x->result()); }
    /// construct a Axpby object
    inline
        Op::op_ptr operator-(Op::op_ptr x)              { return boost::make_shared<Axpby>(x->result(), x->result(), 0.f,-1.f); }
    /// construct a Atan2 object
    inline
        Op::op_ptr atan2(Op::op_ptr y, Op::op_ptr x)    { return boost::make_shared<Atan2>(y->result(), x->result()); }
    /// construct a Log object
    inline
        Op::op_ptr log(Op::op_ptr x)                    { return boost::make_shared<Log>(x->result()); }
    /// construct a Pow object
    inline
        Op::op_ptr square(Op::op_ptr x)                 { return boost::make_shared<Pow>(2.f, x->result()); }
    /// construct a Pow object
    inline
        Op::op_ptr sqrt(Op::op_ptr x)                   { return boost::make_shared<Pow>(.5f, x->result()); }
    /// construct a Pow object
    inline
        Op::op_ptr pow(Op::op_ptr x, float f)           { return boost::make_shared<Pow>(f, x->result()); }
    /// construct a Abs object
    inline
        Op::op_ptr abs(Op::op_ptr x, float eps=0.0001f) { return boost::make_shared<Abs>(x->result(), eps); }
    /// construct a Exp object
    inline
        Op::op_ptr exp(float f, Op::op_ptr x)           { return boost::make_shared<Exp>(f, x->result()); }
    /// construct a Transpose object
    inline
        Op::op_ptr transpose(Op::op_ptr x) { return boost::make_shared<Transpose>(x->result()); }
    /// construct a Prod object
    inline
        Op::op_ptr prod(Op::op_ptr x, Op::op_ptr y, char tx='n', char ty='n') { return boost::make_shared<Prod>(x->result(), y->result(), tx, ty); }
    /// construct an input object
    /// @param e an extents object, describing the dimensions of the input
    template<class E>
    inline
        boost::shared_ptr<ParameterInput> input(E e){ return boost::make_shared<ParameterInput>(e); }
    /// construct an input object
    /// @param e an extents object, describing the dimensions of the input
    /// @param name an identifier for visualization
    template<class E>
    inline
        boost::shared_ptr<ParameterInput> input(E e, const std::string& name){ return boost::make_shared<ParameterInput>(e, name); }
    /// construct a Sink object
    inline
        boost::shared_ptr<Sink> sink(Op::op_ptr x, unsigned int res=0){ return boost::make_shared<Sink>(x->result(res)); }
    /// construct a Sink object
    inline
        boost::shared_ptr<Sink> sink(const std::string& name, Op::op_ptr x, unsigned int res=0){ return boost::make_shared<Sink>(name, x->result(res)); }
    /// construct a Sin object
    inline
        Op::op_ptr sin(Op::op_ptr x)                    { return boost::make_shared<Sin>(x->result()); }
    /// construct a Cos object
    inline
        Op::op_ptr cos(Op::op_ptr x)                    { return boost::make_shared<Cos>(x->result()); }
    /// construct a Tanh object
    inline
        Op::op_ptr tanh(Op::op_ptr x)                   { return boost::make_shared<Tanh>(x->result()); }
    /// construct a RectifiedLinear object
    inline
        Op::op_ptr rectified_linear(Op::op_ptr x)       { return boost::make_shared<RectifiedLinear>(x->result()); }
    /// construct a Logistic object
    inline
        Op::op_ptr logistic(Op::op_ptr x)               { return boost::make_shared<Logistic>(x->result()); }
    /// construct a Logistic object
    inline
        Op::op_ptr logistic(Op::op_ptr x, bool b)       { return boost::make_shared<Logistic>(x->result(),b); }
    /// construct a NegCrossEntropyOfLogistic object
    inline
        Op::op_ptr neg_log_cross_entropy_of_logistic(Op::op_ptr x, Op::op_ptr y) { return boost::make_shared<NegCrossEntropyOfLogistic>(x->result(), y->result()); }
    /// construct a Axpby object
    inline
        Op::op_ptr axpby(float a, Op::op_ptr x, float b, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(),y->result(),a,b); }
    /// construct a Axpby object
    inline
        Op::op_ptr axpby(float a, Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(),y->result(),a,1.f); }
    /// construct a Axpby object
    inline
        Op::op_ptr axpby(Op::op_ptr x, float b, Op::op_ptr y){ return boost::make_shared<Axpby>(x->result(),y->result(),1.f,b); }
    /// construct a Noiser object
    inline
        Op::op_ptr add_rnd_normal(Op::op_ptr x, float f){ return boost::make_shared<Noiser>(x->result(),f, Noiser::NT_NORMAL); }
    /// construct a Noiser object
    inline
        Op::op_ptr zero_out(Op::op_ptr x, float f){ return boost::make_shared<Noiser>(x->result(),f, Noiser::NT_ZERO_OUT); }
    /// construct a Sum object
    inline
        Op::op_ptr sum(Op::op_ptr x)                    { return boost::make_shared<Sum>(x->result()); }
    /// construct a SumMatToVec object
    inline
        Op::op_ptr mean_to_vec(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<SumMatToVec>(x->result(), ax, true ); }
    /// construct a SumMatToVec object
    inline
        Op::op_ptr var_to_vec(Op::op_ptr x, unsigned int ax)   { 
            return boost::make_shared<SumMatToVec>(x->result(), ax, true, true )  // mean(x^2)
                - square(mean_to_vec(x, ax)); // mean(x)^2
        }
    /// construct a SumMatToVec object
    inline
        Op::op_ptr sum_to_vec_squared(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<SumMatToVec>(x->result(), ax, false, true ); }
    /// construct a SumMatToVec object
    inline
        Op::op_ptr sum_to_vec(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<SumMatToVec>(x->result(), ax ); }
    /// construct a Mean object
    inline
        Op::op_ptr mean(Op::op_ptr x)                   { return boost::make_shared<Mean>(x->result()); }
    /// return a function object that has a scalar result for the variance
    inline
        Op::op_ptr var(Op::op_ptr x)                   { return mean(square(x)) - square(mean(x)); }
    /// construct a MatPlusVec object
    inline
        Op::op_ptr mat_plus_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax) { return boost::make_shared<MatPlusVec>(x->result(),v->result(), ax); }
    /// construct a MatTimesVec object
    inline
        Op::op_ptr mat_times_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax) { return boost::make_shared<MatTimesVec>(x->result(),v->result(), ax); }
    /// construct a PairwiseNorm object
    inline
        Op::op_ptr pairwise_norm(Op::op_ptr img, unsigned int dim) { return boost::make_shared<PairwiseNorm>(img->result(), dim); }
    /// construct a Convolve object
    inline
        Op::op_ptr convolve(Op::op_ptr img, Op::op_ptr flt, bool padding=false, int partialSum=4) { return boost::make_shared<Convolve>(img->result(),flt->result(), padding, partialSum); }
#ifndef NO_THEANO_WRAPPERS
    /// construct a Convolve theano object
    inline
        Op::op_ptr convolve2dTheano(Op::op_ptr img, Op::op_ptr flt, std::string mode = "valid") { return boost::make_shared<Convolve2dTheano>(img->result(),flt->result(), mode); }
#endif
    /// construct a ReorderForConv object
    inline
        Op::op_ptr reorder_for_conv(Op::op_ptr img) { return boost::make_shared<ReorderForConv>(img->result()); }
    /// construct a ReorderFromConv object
    inline
        Op::op_ptr reorder_from_conv(Op::op_ptr img) { return boost::make_shared<ReorderFromConv>(img->result()); }
    /// construct a ContrastNormalization object
    inline
        Op::op_ptr contrast_normalization(Op::op_ptr img, int patchSize, float addScale, float powScale) { return boost::make_shared<ContrastNormalization>(img->result(), patchSize, addScale, powScale); }
    /// construct a ResponseNormalizationCrossMaps object
    inline
        Op::op_ptr response_normalization_cross_maps(Op::op_ptr img, int groups, float addScale=0.0000125f, float powScale=0.75f, bool blocked=false) { return boost::make_shared<ResponseNormalizationCrossMaps>(img->result(), groups, addScale, powScale, blocked); }
    /// construct a ResponseNormalization object
    inline
        Op::op_ptr response_normalization(Op::op_ptr img, int patchSize, float addScale, float powScale) { return boost::make_shared<ResponseNormalization>(img->result(), patchSize, addScale=0.0000125f, powScale=0.5f); }
    /// construct a BedOfNails object
    inline
        Op::op_ptr bed_of_nails(Op::op_ptr img, int stridex=2, int startx=0) { return boost::make_shared<BedOfNails>(img->result(), stridex, startx); }
    /// construct a ResizeBilinear object
    inline
        Op::op_ptr resize_bilinear(Op::op_ptr img, float scale) { return boost::make_shared<ResizeBilinear>(img->result(), scale); }
    /// construct a SeparableFilter object
    inline
        Op::op_ptr separable_filter(Op::op_ptr img, const matrix& kernel) { return boost::make_shared<SeparableFilter>(img->result(), kernel); }
    /// construct a Flatten object
    inline
        Op::op_ptr flatten(Op::op_ptr img, unsigned int outdim=1) { return boost::make_shared<Flatten>(img->result(),outdim); }
    /// construct a LocalPooling object
    inline
        Op::op_ptr local_pool(Op::op_ptr img, cuv::alex_conv::pool_type pt) { return boost::make_shared<LocalPooling>(img->result(),pt); }

    /// construct a Reshape object
    template<std::size_t D>
    inline
        Op::op_ptr reshape(Op::op_ptr img, const cuv::extent_gen<D>& eg) { return boost::make_shared<Reshape>(img->result(),eg); }

    /// construct a Subtensor object
    template<std::size_t D, std::size_t E>
    inline
        Op::op_ptr subtensor(Op::op_ptr img, const cuv::index_gen<D,E>& idx, bool copy) { return boost::make_shared<Subtensor>(img->result(),idx, copy); }

    /// construct a Concatenate object
    inline
        Op::op_ptr concatenate(Op::op_ptr img1, Op::op_ptr img2, unsigned int dim) { return boost::make_shared<Concatenate>(img1->result(), img2->result(), dim); }

    /// construct a Softmax object
    inline
        Op::op_ptr softmax(Op::op_ptr img, unsigned int dim=0){ return boost::make_shared<Softmax>(img->result(), dim); }
    /// construct a MultinomialLogisticLoss object
    inline
        Op::op_ptr multinomial_logistic_loss(Op::op_ptr x, Op::op_ptr target, unsigned int dim=0){ return boost::make_shared<MultinomialLogisticLoss>(x->result(), target->result(), dim); }
    /// construct a EpsilonInsensitiveLoss object
    inline
        Op::op_ptr epsilon_insensitive_loss(float sensitivity, Op::op_ptr target, Op::op_ptr x){ return boost::make_shared<EpsilonInsensitiveLoss>(sensitivity, target->result(), x->result()); }
    /// construct a RowSelector object
    inline
        boost::shared_ptr<RowSelector> row_select(Op::op_ptr p0, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), row); }
    /// construct a RowSelector object
    inline
        boost::shared_ptr<RowSelector> row_select(Op::op_ptr p0, Op::op_ptr p1, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), p1->result(), row); }
    /// construct a RowSelector object
    inline
        boost::shared_ptr<RowSelector> row_select(Op::op_ptr p0, Op::op_ptr p1, Op::op_ptr p2, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), p1->result(), p2->result(), row); }
    /// construct a RowSelector object
    inline
        boost::shared_ptr<RowSelector> row_select(Op::op_ptr p0, Op::op_ptr p1, Op::op_ptr p2, Op::op_ptr p3, int row=-1){ return boost::make_shared<RowSelector>(p0->result(), p1->result(), p2->result(), p3->result(), row); }
    /// construct a RowRangeSelector object
    inline
        boost::shared_ptr<RowRangeSelector> row_range_select(Op::op_ptr p0, int n_rows, int row=-1){ return boost::make_shared<RowRangeSelector>(p0->result(), n_rows, row); }
    /// construct a RowRangeSelector object
    inline
        boost::shared_ptr<RowRangeSelector> row_range_select(Op::op_ptr p0, Op::op_ptr p1, int n_rows, int row=-1){ return boost::make_shared<RowRangeSelector>(p0->result(), p1->result(), n_rows, row); }
    /// construct a Pipe object
    inline 
        Op::op_ptr result(Op::op_ptr p0, unsigned int result=0){ return boost::make_shared<Pipe>(p0->result(result), result); }

    /// construct a ClassificationLoss object
    inline
        Op::op_ptr classification_loss(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<ClassificationLoss>(x->result(),y->result()); }
    /// construct a F2Measure object (including binary confusion matrix)
    inline
        Op::op_ptr f2_measure(Op::op_ptr tch, Op::op_ptr res, float thresh_tch, float thresh_res){ return boost::make_shared<F2Measure>(tch->result(),res->result(), thresh_tch, thresh_res); }
    /// construct a F2Measure object (including binary confusion matrix)
    inline
        Op::op_ptr f2_measure(Op::op_ptr tch, Op::op_ptr res, Op::op_ptr ign, float thresh_tch, float thresh_res){ return boost::make_shared<F2Measure>(tch->result(),res->result(), ign->result(), thresh_tch, thresh_res); }
    /// construct a ClassificationLoss object
    inline
        Op::op_ptr classification_loss(boost::shared_ptr<Sink> x, Op::op_ptr y){ return boost::make_shared<ClassificationLoss>(x->result(),y->result()); }
    
    /// construct a HingeLoss object
    inline
        Op::op_ptr hinge_loss(Op::op_ptr y, Op::op_ptr y_hat){ return boost::make_shared<HingeLoss>(y->result(),y_hat->result(), false /* not squared */); }

    /// construct a HingeLoss object
    inline
        Op::op_ptr squared_hinge_loss(Op::op_ptr y, Op::op_ptr y_hat){ return boost::make_shared<HingeLoss>(y->result(),y_hat->result(), true /* squared */); }

    /// adds to the value of another Op's parameter
    inline 
        Op::op_ptr add_to_param(Op::op_ptr dst, Op::op_ptr src, int param=0, int result=0){
            dst->param(param)->param_uses.push_back(src->result(result));
            src->result(result)->result_uses.push_back(dst->param(param));
            return dst;
        }
    /// construct a DeltaSink object attached to a \c op_param.
    /// it assumes that the result is only used \b once.
    inline
        boost::shared_ptr<DeltaSink> delta_sink(const std::string& name, Op::op_ptr x, unsigned int res=0){ 
            cuvAssert(x->result(res)->result_uses.size()==1);

            boost::shared_ptr<Op> dst 
                = x->result(res)->result_uses[0].lock()->get_op()->shared_from_this();
            int param_number = x->result(res)->result_uses[0].lock()->param_number;

            boost::shared_ptr<DeltaSink> snk = boost::make_shared<DeltaSink>(name); 
            add_to_param(dst, snk, param_number, 0);
            return snk;
        }
    /// @}

    /// annotate the given operator with a label
    inline
        Op::op_ptr label(const std::string& l, Op::op_ptr x){ x->set_label(l); return x; }
    /// write statistics to log whenever fprop/bprop is called
    inline
        Op::op_ptr printer(const std::string& l, Op::op_ptr x, bool fprop=true, bool bprop=true){ return boost::make_shared<Printer>(l,x->result(), fprop, bprop); }
}
#endif /* __OPS_HPP__ */
