#ifndef __OPS_HPP__
#     define __OPS_HPP__

#include <cmath>
#include <cuvnet/ops/axpby.hpp>
#include <cuvnet/ops/ones_and_zeros.hpp>
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
#include <cuvnet/ops/sum_out_dim.hpp>
#include <cuvnet/ops/log_add_exp.hpp>
#include <cuvnet/ops/upscale.hpp>

#ifndef NO_THEANO_WRAPPERS
#include <cuvnet/ops/theano_ops.hpp>
#endif
namespace cuvnet
{
    /// @addtogroup convenience_funcs
    /// Convenience functions for combining Ops to create more complex Ops.
    /// @{
    
    /// construct a Multiply object
    inline
        Op::op_ptr operator*(Op::op_ptr x, Op::op_ptr y){ return boost::make_shared<Multiply>(x->result(), y->result()); }
    /// construct a ScalarLike object
    inline
        Op::op_ptr ones_like(Op::op_ptr x){ return boost::make_shared<ScalarLike>(x->result(), 1.f); }
    /// construct a ScalarLike object
    inline
        Op::op_ptr zeros_like(Op::op_ptr x){ return boost::make_shared<ScalarLike>(x->result(), 0.f); }
    /// construct a ScalarLike object
    inline
        Op::op_ptr scalar_like(Op::op_ptr x, float f){ return boost::make_shared<ScalarLike>(x->result(), f); }
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
    /// construct a logaddexp object
    inline
        Op::op_ptr log_add_exp(Op::op_ptr x, float f=0.f)           { return boost::make_shared<LogAddExp>(f, x->result()); }        
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
        Op::op_ptr rectified_linear(Op::op_ptr x, bool mem_optimized)       { return boost::make_shared<RectifiedLinear>(x->result(), mem_optimized); }
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
        boost::shared_ptr<Noiser> add_rnd_normal(Op::op_ptr x, float f){ return boost::make_shared<Noiser>(x->result(),f, Noiser::NT_NORMAL); }
    /// construct a Noiser object
    inline
        boost::shared_ptr<Noiser> zero_out(Op::op_ptr x, float f, bool compensate=true){ return boost::make_shared<Noiser>(x->result(),f, Noiser::NT_ZERO_OUT, compensate); }
    /// construct a Noiser object
    inline
        boost::shared_ptr<Noiser> salt_and_pepper(Op::op_ptr x, float f){ return boost::make_shared<Noiser>(x->result(),f, Noiser::NT_SALT_AND_PEPPER, false); }
    /// construct a Sum object
    inline
        Op::op_ptr sum(Op::op_ptr x)                    { return boost::make_shared<Sum>(x->result()); }
    /// construct a SumMatToVec object
    inline
        Op::op_ptr mean_to_vec(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<SumMatToVec>(x->result(), ax, true ); }
    /// construct a SumMatToVec object
    inline
        Op::op_ptr mean_to_vec_squared(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<SumMatToVec>(x->result(), ax, true, true ); }
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

    /// construct a sum object (python style)
    inline
        Op::op_ptr sum(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<Sum_Out_Dim>(x->result(), ax, false, false ); }
    /// construct a mean object (python style)
    inline
        Op::op_ptr mean(Op::op_ptr x, unsigned int ax)   { return boost::make_shared<Sum_Out_Dim>(x->result(), ax, true, false ); }

        /// construct a Mean object
    inline
        Op::op_ptr mean(Op::op_ptr x)                   { return boost::make_shared<Mean>(x->result()); }
    /// return a function object that has a scalar result for the variance
    inline
        Op::op_ptr var(Op::op_ptr x)                   { return mean(square(x)) - square(mean(x)); }
    /// construct a MatPlusVec object
    inline
        Op::op_ptr mat_plus_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax, bool subtract_mean=false) { return boost::make_shared<MatPlusVec>(x->result(),v->result(), ax, subtract_mean); }
    /// construct a MatTimesVec object
    inline
        Op::op_ptr mat_times_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax) { return boost::make_shared<MatTimesVec>(x->result(),v->result(), ax); }
    /// construct a MatDivideVec object
    inline
        Op::op_ptr mat_divide_vec(Op::op_ptr x, Op::op_ptr v, unsigned int ax) { return boost::make_shared<MatDivideVec>(x->result(),v->result(), ax); }
    /// construct a Tuplewise_op object
    inline
        Op::op_ptr tuplewise_op(Op::op_ptr img, unsigned int dim, unsigned int sub_size=2, cuv::alex_conv::tuplewise_op_functor to = cuv::alex_conv::TO_NORM, float epsilon=0.f) { return boost::make_shared<Tuplewise_op>(img->result(), dim, sub_size, to, epsilon); }
    /// construct a Convolve object
    inline
        boost::shared_ptr<Convolve> convolve(Op::op_ptr img, Op::op_ptr flt, bool padding, int padding_size, int stride, int ngroups, int partialSum=4) { return boost::make_shared<Convolve>(img->result(),flt->result(), padding, padding_size, stride, ngroups, partialSum); }
#ifndef NO_THEANO_WRAPPERS
    /// construct a Convolve theano object
    inline
        Op::op_ptr convolve2dTheano(Op::op_ptr img, Op::op_ptr flt, std::string mode = "valid", Op::op_ptr bias = Op::op_ptr()) { 
            if (bias)
                return boost::make_shared<Convolve2dTheano>(img->result(),flt->result(), bias->result(), mode);
            else 
                return boost::make_shared<Convolve2dTheano>(img->result(),flt->result(), mode);
        }
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
    /// construct a SeparableFilter1d object
    inline
        Op::op_ptr separable_filter1d(Op::op_ptr img, const cuv::tensor<float,cuv::host_memory_space>& kernel, unsigned int dim = 0) { return boost::make_shared<SeparableFilter1d>(img->result(), kernel, dim); }
    /// construct a Flatten object
    inline
        Op::op_ptr flatten(Op::op_ptr img, unsigned int outdim=1, bool copy=true) { return boost::make_shared<Flatten>(img->result(),outdim, copy); }
    /// construct a LocalPooling object
    inline
        //Op::op_ptr local_pool(Op::op_ptr img, int subsx, int stridex, cuv::alex_conv::pool_type pt) { return boost::make_shared<LocalPooling>(img->result(), subsx, stridex, pt, (subsx%2==0?0:subsx/-2)); }
        Op::op_ptr local_pool(Op::op_ptr img, int subsx, int stridex, cuv::alex_conv::pool_type pt) { return boost::make_shared<LocalPooling>(img->result(), subsx, stridex, pt, 0); }

    /// construct a Reshape object
    template<std::size_t D>
    inline
        Op::op_ptr reshape(Op::op_ptr img, const cuv::extent_gen<D>& eg, bool copy=true) { return boost::make_shared<Reshape>(img->result(),eg, copy); }


#ifndef NO_THEANO_WRAPPERS
    /// construct a ShuffleDim object
    template<std::size_t D>
    inline
       Op::op_ptr shuffle_dim(Op::op_ptr img, const cuv::extent_gen<D>& eg) { return boost::make_shared<ShuffleDim>(img->result(),eg); }

    /// construct a FlipDims object
    template<std::size_t D>
    inline
       Op::op_ptr flip_dims(Op::op_ptr img, const cuv::extent_gen<D>& eg) { return boost::make_shared<FlipDims>(img->result(), eg); }
#endif

    /// construct a Subtensor object
    //template<std::size_t D, std::size_t E>
    template<int D, int E>
    inline
        Op::op_ptr subtensor(Op::op_ptr img, const cuv::index_gen<D,E>& idx, bool copy) { return boost::make_shared<Subtensor>(img->result(),idx, copy); }

    /// construct a Concatenate object
    inline
        Op::op_ptr concatenate(Op::op_ptr img1, Op::op_ptr img2, unsigned int dim) { 
             boost::shared_ptr<std::vector<Op::result_t> > res(new std::vector<Op::result_t>(2));
             (*res)[0] = img1->result();
             (*res)[1] = img2->result();
            return boost::make_shared<Concatenate>( res, dim); }

    /// construct a Concatenate_n object, which concatenates n input matrices
    inline
        Op::op_ptr concatenate(std::vector<Op::op_ptr> in, unsigned int dim) { 
            unsigned int size = in.size();
             boost::shared_ptr<std::vector<Op::result_t> > res(new std::vector<Op::result_t>(size));
            for ( unsigned int i = 0; i < size; i++) (*res)[i] = in[i]->result();
            return boost::make_shared<Concatenate>( res, dim); 
        }
        
        
    /// construct a Softmax object
    inline
        Op::op_ptr softmax(Op::op_ptr img, unsigned int dim=0){ return boost::make_shared<Softmax>(img->result(), dim); }
    /// construct a MultinomialLogisticLoss object
    inline
        Op::op_ptr multinomial_logistic_loss(Op::op_ptr x, Op::op_ptr target, unsigned int dim=0){ return boost::make_shared<MultinomialLogisticLoss>(x->result(), target->result(), dim); }
    /// construct a MultinomialLogisticLoss2 object, which should be more
    /// efficient than MultinomialLogisticLoss, but outputs only a single number for the whole batch
    inline
        Op::op_ptr multinomial_logistic_loss2(Op::op_ptr x, Op::op_ptr target, unsigned int pattern_axis=0){ return boost::make_shared<MultinomialLogisticLoss2>(x->result(), target->result(), pattern_axis); }
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
        Op::op_ptr result(Op::op_ptr p0, unsigned int result=0){ 
            if(result == 0)
                return p0;
            return boost::make_shared<Pipe>(p0->result(result), result); 
        }

    /// construct a ClassificationLoss object
    inline
        Op::op_ptr classification_loss(Op::op_ptr x, Op::op_ptr y, int axis=0){ return boost::make_shared<ClassificationLoss>(x->result(),y->result(),axis); }
    inline
        Op::op_ptr classification_loss(Op::op_ptr x, Op::op_ptr y, Op::op_ptr z, unsigned int axis=0){ return boost::make_shared<ClassificationLoss>(x->result(),y->result(),z->result(),axis); }
    /// construct a F2Measure object (including binary confusion matrix)
    inline
        Op::op_ptr f2_measure(Op::op_ptr tch, Op::op_ptr res, float thresh_tch, float thresh_res){ return boost::make_shared<F2Measure>(tch->result(),res->result(), thresh_tch, thresh_res); }
    /// construct a F2Measure object (including binary confusion matrix)
    inline
        Op::op_ptr f2_measure(Op::op_ptr tch, Op::op_ptr res, Op::op_ptr ign, float thresh_tch, float thresh_res){ return boost::make_shared<F2Measure>(tch->result(),res->result(), ign->result(), thresh_tch, thresh_res); }
    /// construct a ClassificationLoss object
    inline
        Op::op_ptr classification_loss(boost::shared_ptr<Sink> x, Op::op_ptr y, unsigned int axis=0){ return boost::make_shared<ClassificationLoss>(x->result(),y->result(), axis); }
    inline
        Op::op_ptr classification_loss(boost::shared_ptr<Sink> x, Op::op_ptr y, Op::op_ptr z, unsigned int axis=0){ return boost::make_shared<ClassificationLoss>(x->result(),y->result(),z->result(), axis); }
    
    /// construct a HingeLoss object
    inline
        Op::op_ptr hinge_loss(Op::op_ptr y, Op::op_ptr y_hat){ return boost::make_shared<HingeLoss>(y->result(),y_hat->result(), false /* not squared */); }

    /// construct a HingeLoss object
    inline
        Op::op_ptr squared_hinge_loss(Op::op_ptr y, Op::op_ptr y_hat){ return boost::make_shared<HingeLoss>(y->result(),y_hat->result(), true /* squared */); }

    /**
     * adds to the value of another Op's parameter
     *
     * @param dst the op whose parameter we want to add to
     * @param src the op which we want to add
     * @param param the index of the parameter of dst
     * @param result the index of the result of src
     */
    inline 
        Op::op_ptr add_to_param(Op::op_ptr dst, Op::op_ptr src, int param=0, int result=0){
            dst->param(param)->param_uses.push_back(src->result(result));
            src->result(result)->result_uses.push_back(dst->param(param));
            return dst;
        }
    /** 
     * construct a DeltaSink object attached to a \c op_param.
     * 
     * @param name a arbitrary identifier for the sink (for visualization mostly)
     * @param x the operater whose delta we'd like to record
     * @param param the index of the parameter of x we'd like to record
     */
    inline
        boost::shared_ptr<DeltaSink> 
        delta_sink(const std::string& name, Op::op_ptr x, unsigned int param=0){ 
            boost::shared_ptr<DeltaSink> snk = boost::make_shared<DeltaSink>(name); 
            add_to_param(x, snk, param, 0);
            return snk;
        }
    /// @}

    /// annotate the given operator with a label
    inline
        Op::op_ptr label(const std::string& l, Op::op_ptr x){ x->set_label(l); return x; }
    /// write statistics to log whenever fprop/bprop is called
    
    /// construct an Upscale object
    inline
        Op::op_ptr upscale(Op::op_ptr img, unsigned int factor) { 
             //boost::shared_ptr<std::vector<Op::result_t> > res(new std::vector<Op::result_t>(2));
             //(*res)[0] = img1->result();
             //(*res)[1] = img2->result();
            return boost::make_shared<Upscale>( img->result(), factor); }
   inline
   Op::op_ptr printer(const std::string& l, Op::op_ptr x, bool fprop=true, bool bprop=true){ return boost::make_shared<Printer>(l,x->result(), fprop, bprop); }
}
#endif /* __OPS_HPP__ */
