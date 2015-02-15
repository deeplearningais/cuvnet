#include <boost/scope_exit.hpp>
#include <cuvnet/common.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops/cudnn.hpp>
#include <third_party/cudnn-6.5-linux-R1/cudnn.h>
//#include <third_party/cudnn-6.5-linux-x64-R2-rc1/cudnn.h>
#include </home/stud/sheikhr/cuda-workspace/streamsHelper/kernels.hpp>

#define DIVUP(x,y) (((x)+ (y) -1) / (y))
#define CUDNN_CALL(XXX) if(1){ \
    cudnnStatus_t status = XXX; \
    if (status != CUDNN_STATUS_SUCCESS) \
        throw(std::string("ERROR ") + #XXX + ", status: " + boost::lexical_cast<std::string>(status));}

#ifndef CUDNN_VERSION
// must be version 1
#define cudnnTensorDescriptor_t cudnnTensor4dDescriptor_t
#define cudnnCreateTensorDescriptor cudnnCreateTensor4dDescriptor
#define cudnnDestroyTensorDescriptor cudnnDestroyTensor4dDescriptor
#define cudnnSetFilter4dDescriptor cudnnSetFilterDescriptor
#define cudnnAddTensor cudnnAddTensor4d
#define CUDNN_VERSION 1
#endif
    
namespace
{
    log4cxx::LoggerPtr g_log(log4cxx::Logger::getLogger("convolve_cudnn")); 
}


namespace cuvnet
{
    /***************************************************
     * ConvolvecuDNN
     ***************************************************/

	template<class T>
	cudnnDataType_t cudnn_data_type() {
		if (cuv::IsSame<typename T::value_type, float>::Result::value)
			return CUDNN_DATA_FLOAT;
		if (cuv::IsSame<typename T::value_type, double>::Result::value)
			return CUDNN_DATA_DOUBLE;
		throw std::runtime_error("CUDNN data type unavailable");
	}

    struct cudnn_state{
		cudnnHandle_t handle1, handle2, handle3;
		cudnnTensorDescriptor_t imgDesc, outputDesc, biasDesc,
                                gradOutputDesc, gradImgDesc, gradBiasDesc;
        cudnnFilterDescriptor_t filterDesc, gradFilterDesc;
        cudnnConvolutionDescriptor_t convDesc;
        cudaStream_t stream1, stream2, stream3;
#if CUDNN_VERSION != 1
        cudnnConvolutionFwdAlgo_t algo;
#endif
        cudnnAddMode_t add_mode;
        size_t workspace_size;
        cudnn_state(ConvolvecuDNN* conv,
                std::vector<unsigned int> img_shape, 
                std::vector<unsigned int> filter_shape,
                std::vector<unsigned int> out_shape,
                std::vector<unsigned int> bias_shape
                ){

            createCudaStream(stream1);
        	createCudaStream(stream2);
        	createCudaStream(stream3);

        	CUDNN_CALL(cudnnCreate(&handle1));
        	CUDNN_CALL(cudnnCreate(&handle2));
        	CUDNN_CALL(cudnnCreate(&handle3));

        	CUDNN_CALL(cudnnSetStream(handle1, stream1));
        	CUDNN_CALL(cudnnSetStream(handle2, stream2));
        	CUDNN_CALL(cudnnSetStream(handle3, stream3));

            CUDNN_CALL(cudnnCreateTensorDescriptor(&imgDesc));
            CUDNN_CALL(cudnnCreateTensorDescriptor(&outputDesc));
            CUDNN_CALL(cudnnCreateTensorDescriptor(&biasDesc));
            CUDNN_CALL(cudnnCreateFilterDescriptor(&filterDesc));

            CUDNN_CALL(cudnnCreateTensorDescriptor(&gradImgDesc));
            CUDNN_CALL(cudnnCreateTensorDescriptor(&gradOutputDesc));
            CUDNN_CALL(cudnnCreateTensorDescriptor(&gradBiasDesc));
            CUDNN_CALL(cudnnCreateFilterDescriptor(&gradFilterDesc));

            cudnnDataType_t dtype = cudnn_data_type<matrix>();

            // Set descriptors
            CUDNN_CALL(cudnnSetTensor4dDescriptor(imgDesc, CUDNN_TENSOR_NCHW, dtype, img_shape[0], img_shape[1], img_shape[2], img_shape[3]));
            CUDNN_CALL(cudnnSetTensor4dDescriptor(gradImgDesc, CUDNN_TENSOR_NCHW, dtype, img_shape[0], img_shape[1], img_shape[2], img_shape[3]));
            if(bias_shape.size()){
                CUDNN_CALL(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, dtype, bias_shape[0], bias_shape[1], bias_shape[2], bias_shape[3]));
                CUDNN_CALL(cudnnSetTensor4dDescriptor(gradBiasDesc, CUDNN_TENSOR_NCHW, dtype, bias_shape[0], bias_shape[1], bias_shape[2], bias_shape[3]));
            }
            CUDNN_CALL(cudnnSetFilter4dDescriptor(filterDesc, dtype, filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]));
            CUDNN_CALL(cudnnSetFilter4dDescriptor(gradFilterDesc, dtype, filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]));
            CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
#if CUDNN_VERSION == 1
            CUDNN_CALL(cudnnSetConvolutionDescriptor(convDesc, imgDesc, filterDesc, conv->m_padding_y, conv->m_padding_x, conv->m_ver_filt_stride, conv->m_hor_filt_stride, 1, 1, CUDNN_CONVOLUTION));
#else
            CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc, conv->m_padding_y, conv->m_padding_x, conv->m_ver_filt_stride, conv->m_hor_filt_stride, 1, 1, CUDNN_CONVOLUTION));
#endif

            if(0);
            else if(bias_shape.size() == 0);
            else if(bias_shape[2] == 1) add_mode = CUDNN_ADD_SAME_C;       // a (set of) 1x1 images
            else if(bias_shape[1] == 1) add_mode = CUDNN_ADD_FEATURE_MAP;  // a single image
            else add_mode = CUDNN_ADD_FULL_TENSOR;                         // same size as input

            // query output layout
            int n_out, c_out, h_out, w_out;
#if CUDNN_VERSION == 1
            CUDNN_CALL(cudnnGetOutputTensor4dDim(convDesc, CUDNN_CONVOLUTION_FWD, &n_out, &c_out, &h_out, &w_out));
#else
            CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convDesc, imgDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
#endif
            //std::cout<<"cuddn "<<h_out<<", detShapes "<<out_shape[2]<<std::endl;
            cuvAssert((unsigned)n_out == out_shape[0]);
            cuvAssert((unsigned)c_out == out_shape[1]);
            cuvAssert((unsigned)h_out == out_shape[2]);
            cuvAssert((unsigned)w_out == out_shape[3]);


            // Set and allocate output tensor descriptor
            CUDNN_CALL(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, dtype, n_out, c_out, h_out, w_out));
            CUDNN_CALL(cudnnSetTensor4dDescriptor(gradOutputDesc, CUDNN_TENSOR_NCHW, dtype, n_out, c_out, h_out, w_out));

#if CUDNN_VERSION != 1
            CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
                    handle,
                    imgDesc,
                    filterDesc,
                    convDesc,
                    outputDesc,
                    //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                    0,
                    &algo));
            CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle,
                        imgDesc, filterDesc, convDesc, outputDesc, algo,
                        &workspace_size));
#endif
        }
        ~cudnn_state(){
            CUDNN_CALL(cudnnDestroyTensorDescriptor(imgDesc));
            CUDNN_CALL(cudnnDestroyTensorDescriptor(gradImgDesc));

            CUDNN_CALL(cudnnDestroyTensorDescriptor(outputDesc));
            CUDNN_CALL(cudnnDestroyTensorDescriptor(gradOutputDesc));

            CUDNN_CALL(cudnnDestroyFilterDescriptor(filterDesc));
            CUDNN_CALL(cudnnDestroyFilterDescriptor(gradFilterDesc));

            CUDNN_CALL(cudnnDestroyTensorDescriptor(biasDesc));
            CUDNN_CALL(cudnnDestroyTensorDescriptor(gradBiasDesc));

            CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));

            destroyCudaStream(stream1);
            destroyCudaStream(stream2);
            destroyCudaStream(stream3);
            CUDNN_CALL(cudnnDestroy(handle1));
            CUDNN_CALL(cudnnDestroy(handle2));
            CUDNN_CALL(cudnnDestroy(handle3));
        }
    };

	void ConvolvecuDNN::fprop() {

		using namespace cuv;
		using namespace std;

		param_t::element_type& p0 = *m_params[0];
		param_t::element_type& p1 = *m_params[1];
		result_t::element_type& r0 = *m_results[0];
		cuvAssert(p0.value.cdata().is_c_contiguous());
        //log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("convolve"));
        //LOG4CXX_WARN(log, "CONV fs"<<p1.shape[2] << " srcM"<<p1.shape[1] << " dstM"<<p1.shape[0]<<" pad"<<m_padding_y);

#if CUDNN_VERSION != 1
        cuv::tensor<unsigned char,matrix::memory_space_type>  workspace(m_state->workspace_size, Op::value_ptr::s_allocator);
#endif

		const matrix::value_type alpha = 1.0;
        
        const matrix::value_type* imgData = p0.value.cdata().ptr();
        const matrix::value_type* filterData = p1.value.cdata().ptr();

		if (r0.can_overwrite_directly() || r0.can_add_directly()) {
            const matrix::value_type beta = r0.can_add_directly() ? 1.0 : 0.0;

			matrix::value_type* outputData = r0.overwrite_or_add_value()->ptr();
			// launch convolution on GPU
#if CUDNN_VERSION != 1
			CUDNN_CALL(cudnnConvolutionForward(m_state->handle1, &alpha,
                    m_state->imgDesc, imgData,
                    m_state->filterDesc, filterData,
                    m_state->convDesc, m_state->algo, workspace.ptr(), m_state->workspace_size,
                    &beta,
                    m_state->outputDesc, outputData));
#else
            cudnnConvolutionForward(m_state->handle1, m_state->imgDesc, imgData, m_state->filterDesc,
                    filterData, m_state->convDesc, m_state->outputDesc, outputData,
                    beta == 1 ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE);
#endif
            if(m_params.size() == 3){
#if CUDNN_VERSION != 1
                const float beta = 1.f;
                CUDNN_CALL(cudnnAddTensor(m_state->handle1, m_state->add_mode, &alpha, m_state->biasDesc,
                            m_params[2]->value.cdata().ptr(), &beta,
                            m_state->outputDesc, outputData));
#else
                CUDNN_CALL(cudnnAddTensor(m_state->handle1, m_state->add_mode, &alpha, m_state->biasDesc,
                            m_params[2]->value.cdata().ptr(),
                            m_state->outputDesc, outputData));
#endif
            }

		} else {
			// reallocate *sigh*
            const matrix::value_type beta = 0.;
			value_ptr v(new value_type(r0.shape, cuvnet::get_global_allocator()));
			matrix::value_type* outputData = v->ptr();
#if CUDNN_VERSION != 1
			CUDNN_CALL(cudnnConvolutionForward(m_state->handle1, &alpha,
                    m_state->imgDesc, imgData,
                    m_state->filterDesc, filterData,
                    m_state->convDesc, m_state->algo, workspace.ptr(), m_state->workspace_size,
                    &beta,
                    m_state->outputDesc, outputData));
#else
            cudnnConvolutionForward(m_state->handle1, m_state->imgDesc, imgData, m_state->filterDesc,
                    filterData, m_state->convDesc, m_state->outputDesc, outputData,
                    CUDNN_RESULT_NO_ACCUMULATE);
#endif
            if(m_params.size() == 3){
#if CUDNN_VERSION != 1
                const float beta = 1.0;
                CUDNN_CALL(cudnnAddTensor(m_state->handle1, m_state->add_mode, &alpha, m_state->biasDesc,
                            m_params[2]->value.cdata().ptr(), &beta,
                            m_state->outputDesc, outputData));
#else
                CUDNN_CALL(cudnnAddTensor(m_state->handle1, m_state->add_mode, &alpha, m_state->biasDesc,
                            m_params[2]->value.cdata().ptr(),
                            m_state->outputDesc, outputData));
#endif
            }
            //log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("convolve"));
            //LOG4CXX_WARN(log, "max flt: " << cuv::maximum(p1.value.cdata()));
            //LOG4CXX_WARN(log, "max  in: " << cuv::maximum(p0.value.cdata()));
            //LOG4CXX_WARN(log, "min  in: " << cuv::minimum(p0.value.cdata()));
            //LOG4CXX_WARN(log, "max out: " << cuv::maximum(*v));
			r0.push(v);
		}
		if (!p0.need_derivative && !p1.need_derivative) {
			p0.value.reset();
			p1.value.reset();
		}
	}

	void ConvolvecuDNN::bprop() {
		using namespace cuv;
		using namespace std;

		param_t::element_type& p0 = *m_params[0];
		param_t::element_type& p1 = *m_params[1];
		result_t::element_type& r0 = *m_results[0];

		assert(p0.need_derivative || p1.need_derivative);
		cuvAssert(r0.delta.cdata().is_c_contiguous());

		value_ptr ptr1, ptr2, ptr0;

		if (p1.need_derivative) {
			// calculate p1 first, then we don't need activations
			// anymore and can overwrite them. They are usually
			// larger than the weights, so it should be better in this order.

			const matrix::value_type* imgData = p0.value.cdata().ptr();   //images
			const matrix::value_type* diffData = r0.delta.cdata().ptr();

            const matrix::value_type alpha = 1.0;

			if (p1.can_overwrite_directly() || p1.can_add_directly()) {
                const matrix::value_type beta = p1.can_add_directly() ? 1.0 : 0.0; 
				matrix::value_type* gradFilterData = (*p1.overwrite_or_add_value()).ptr();

#if CUDNN_VERSION != 1
				CUDNN_CALL(cudnnConvolutionBackwardFilter(m_state->handle1, &alpha,
                        m_state->imgDesc, imgData,
                        m_state->gradOutputDesc, diffData,
                        m_state->convDesc, &beta,
                        m_state->filterDesc, gradFilterData));
#else
                CUDNN_CALL(cudnnConvolutionBackwardFilter(m_state->handle1, m_state->imgDesc, imgData, m_state->gradOutputDesc, diffData,
                            m_state->convDesc, m_state->gradFilterDesc, gradFilterData,
                            beta == 1 ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE));
#endif
			} else {


                const matrix::value_type beta = 0.;
				//value_ptr ptr(new value_type(p1.shape, value_ptr::s_allocator));
				ptr1 = value_type(p1.shape, value_ptr::s_allocator);

				matrix::value_type* gradFilterData = (*ptr1).ptr();
#if CUDNN_VERSION != 1
				CUDNN_CALL(cudnnConvolutionBackwardFilter(m_state->handle1, &alpha,
                        m_state->imgDesc, imgData,
                        m_state->gradOutputDesc, diffData,
                        m_state->convDesc, &beta,
                        m_state->gradFilterDesc, gradFilterData));
#else
                CUDNN_CALL(cudnnConvolutionBackwardFilter(m_state->handle1, m_state->imgDesc, imgData, m_state->gradOutputDesc, diffData,
                            m_state->convDesc, m_state->gradFilterDesc, gradFilterData,
                            CUDNN_RESULT_NO_ACCUMULATE));
#endif
				//p1.push(ptr1);
			}
		}

        if(m_params.size() == 3 && m_params[2]->need_derivative){
            param_t::element_type& p2 = *m_params[2];

			const matrix::value_type* diffData = r0.delta.cdata().ptr();

            const matrix::value_type alpha = 1.0;
			if (p2.can_overwrite_directly() || p2.can_add_directly()) {
                const matrix::value_type beta = p2.can_add_directly() ? 1.0 : 0.0; 
				matrix::value_type* gradbiasData = (*p2.overwrite_or_add_value()).ptr();
#if CUDNN_VERSION != 1
				CUDNN_CALL(cudnnConvolutionBackwardBias(m_state->handle2, &alpha,
                        m_state->gradOutputDesc, diffData,
                        &beta,
                        m_state->biasDesc, gradbiasData));
#else
				CUDNN_CALL(cudnnConvolutionBackwardBias(m_state->handle2,
                        m_state->gradOutputDesc, diffData,
                        m_state->biasDesc, gradbiasData,
                        beta == 1 ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE));
#endif
			} else {
                const matrix::value_type beta = 0.;
				//value_ptr ptr(new value_type(p2.shape, value_ptr::s_allocator));
                ptr2 = value_type(p2.shape, value_ptr::s_allocator);

				matrix::value_type* gradbiasData = (*ptr2).ptr();
#if CUDNN_VERSION != 1
				CUDNN_CALL(cudnnConvolutionBackwardBias(m_state->handle2, &alpha,
                        m_state->gradOutputDesc, diffData,
                        &beta,
                        m_state->gradBiasDesc, gradbiasData));
#else
				CUDNN_CALL(cudnnConvolutionBackwardBias(m_state->handle2,
                        m_state->outputDesc, diffData,
                        m_state->biasDesc, gradbiasData,
                        CUDNN_RESULT_NO_ACCUMULATE));
#endif
			//	p2.push(ptr2);
			}
        }

		if (p0.need_derivative) {
			// derivative w.r.t. images

			const matrix::value_type* filterData = p1.value.cdata().ptr();
			const matrix::value_type* diffData = r0.delta.cdata().ptr();

            const matrix::value_type alpha = 1.;
			if (p0.can_overwrite_directly() || p0.can_add_directly()) {
                const matrix::value_type beta = p0.can_add_directly() ? 1.0 : 0.0; 

				matrix::value_type* gradImgData = (*p0.overwrite_or_add_value()).ptr();
#if CUDNN_VERSION != 1
				CUDNN_CALL(cudnnConvolutionBackwardData(m_state->handle3, &alpha, m_state->filterDesc, filterData, m_state->outputDesc, diffData, m_state->convDesc, &beta, m_state->imgDesc, gradImgData));
#else
				CUDNN_CALL(cudnnConvolutionBackwardData(m_state->handle3, m_state->filterDesc, filterData, m_state->outputDesc, diffData, m_state->convDesc, m_state->imgDesc, gradImgData,
                        beta == 1 ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE));
#endif
			}
			else {
                const matrix::value_type beta = 0.;
				//value_ptr ptr(new value_type(p0.shape, value_ptr::s_allocator));
				ptr0 = value_type(p0.shape, value_ptr::s_allocator);
				value_type& v = *ptr0;

				matrix::value_type* gradImgData = v.ptr();

#if CUDNN_VERSION != 1
				CUDNN_CALL(cudnnConvolutionBackwardData(m_state->handle3, &alpha,
                            m_state->filterDesc, filterData,
                            m_state->gradOutputDesc, diffData, 
                            m_state->convDesc, &beta,
                            m_state->gradImgDesc, gradImgData));
#else
				CUDNN_CALL(cudnnConvolutionBackwardData(m_state->handle3,
                            m_state->filterDesc, filterData,
                            m_state->gradOutputDesc, diffData, 
                            m_state->convDesc,
                            m_state->gradImgDesc, gradImgData,
                            beta == 1 ? CUDNN_RESULT_ACCUMULATE : CUDNN_RESULT_NO_ACCUMULATE));
#endif

			//	p0.push(ptr0);
			}
		}
		//synchronize
		emptyKernelCall();

		if (ptr1.ptr() != 0)
			p1.push(ptr1);
		if(m_params.size() == 3 && ptr2.ptr() != 0)
			(*m_params[2]).push(ptr2);
		if (ptr0.ptr() != 0)
			p0.push(ptr0);

	/*	ptr1.reset();
		ptr2.reset();
		ptr0.reset();*/

	}

	void ConvolvecuDNN::_determine_shapes() {

		using namespace std;

		//dst       (nImg, nFilt, nOutPixY, nOutPixX)
		//img       (nImg, nImgChan, nImgPiY, nImgPix)
		//filter    (nFilt,nFiltChan, nFiltPiY,nFiltPix)

		assert(m_params[0]->shape.size() == 4);
		assert(m_params[1]->shape.size() == 4);
		std::vector<unsigned int> dst(4);
		const std::vector<unsigned int>& img = m_params[0]->shape;
		const std::vector<unsigned int>& flt = m_params[1]->shape;
		unsigned int nFilt = flt[0];
		unsigned int nImgPixY = img[2];
		unsigned int nImgPixX = img[3];
		unsigned int nFltPixY = flt[2];
		unsigned int nFltPixX = flt[3];

	    //assertion fails when imgPix=6, stride=3, filter=3, nOutPixY should be 2 but outputs 1
	    //or imgPix=9, stride=2, filter=3, nOutPixY should be 4 but outputs 3

#if 0
	    unsigned int nOutPixX = DIVUP(nImgPixX+2*m_padding_x-nFltPixX, m_hor_filt_stride)+1;
        unsigned int nOutPixY = DIVUP(nImgPixY+2*m_padding_y-nFltPixY, m_ver_filt_stride)+1;
        if(m_hor_filt_stride != 1) nOutPixX -= 1;
        if(m_ver_filt_stride != 1) nOutPixY -= 1;
#else
		unsigned int nOutPixX = DIVUP(nImgPixX+2*m_padding_x-nFltPixX+1, m_hor_filt_stride);
		unsigned int nOutPixY = DIVUP(nImgPixY+2*m_padding_y-nFltPixY+1, m_ver_filt_stride);
#endif

		dst[0] = img[0];
		dst[1] = nFilt;
		dst[2] = nOutPixY;
		dst[3] = nOutPixX;
		m_results[0]->shape = dst;
        if(m_params.size() == 3)
            m_state.reset(new cudnn_state(this, m_params[0]->shape, m_params[1]->shape, m_results[0]->shape, m_params[2]->shape));
        else
            m_state.reset(new cudnn_state(this, m_params[0]->shape, m_params[1]->shape, m_results[0]->shape, std::vector<unsigned int>()));

        log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("determine_shapes"));
        LOG4CXX_WARN(log, "Convolving image of shape ("
                << img[0]
                << " x " << img[1]
                << " x " << img[2]
                << " x " << img[3]
                << ") to shape ("
                << dst[0]
                << " x " << dst[1]
                << " x " << dst[2]
                << " x " << img[3]
                << ") using filters of size " << flt[2]
                << " stride: " << m_hor_filt_stride << " pad: " << m_padding_x);
	}
    void ConvolvecuDNN::_graphviz_node_desc(detail::graphviz_node& desc)const{
        desc.color = "chartreuse4";
        if(m_params[0]->shape.size() > 0 && m_params[1]->shape.size() > 0){
            desc.label = "Conv (" +
                boost::lexical_cast<std::string>(m_params[0]->shape[1]) + ":" +
                boost::lexical_cast<std::string>(m_results[0]->shape[1]) + " fs" +
                boost::lexical_cast<std::string>(m_params[1]->shape[2]) + ")";
        }else{
            desc.label = "Conv";
        }
    }

    /***************************************************
     * cuDNN Pooling
     ***************************************************/

    struct cudnn_pooling_state{
        cudnnHandle_t handle;
    	cudnnPoolingDescriptor_t poolingDesc;
    	cudnnTensorDescriptor_t imgDesc, outDesc;

    	cudnn_pooling_state(PoolingcuDNN* pooling, std::vector<unsigned int> img_shape, std::vector<unsigned int> out_shape) {
			cudnnDataType_t dtype = cudnn_data_type<matrix>();
            CUDNN_CALL(cudnnCreate(&handle));

			CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc));

#if CUDNN_VERSION != 1
			CUDNN_CALL(cudnnSetPooling2dDescriptor(poolingDesc, 
                        //pooling->m_mode == cuv::alex_conv::PT_MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 
                        pooling->m_mode == cuv::alex_conv::PT_MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE,
                        pooling->m_window_height, pooling->m_window_width, pooling->m_vertical_pad, pooling->m_horizontal_pad, pooling->m_vertical_stride, pooling->m_horizontal_stride));
#else
			CUDNN_CALL(cudnnSetPoolingDescriptor(poolingDesc, 
                        pooling->m_mode == cuv::alex_conv::PT_MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE,
                        pooling->m_window_height, pooling->m_window_width,
                        pooling->m_vertical_stride, pooling->m_horizontal_stride));
#endif

			CUDNN_CALL(cudnnCreateTensorDescriptor(&imgDesc));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(imgDesc, CUDNN_TENSOR_NCHW, dtype, img_shape[0], img_shape[1], img_shape[2], img_shape[3]));

			CUDNN_CALL(cudnnCreateTensorDescriptor(&outDesc));
			CUDNN_CALL(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, dtype, out_shape[0], out_shape[1], out_shape[2], out_shape[3]));
        }

		 ~cudnn_pooling_state(){
			 CUDNN_CALL(cudnnDestroyTensorDescriptor(imgDesc));
			 CUDNN_CALL(cudnnDestroyTensorDescriptor(outDesc));
			 CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc));
             CUDNN_CALL(cudnnDestroy(handle));
		 }
    };


    void PoolingcuDNN::release_data(){
    	m_result.reset();
        Op::release_data();
    }

    void PoolingcuDNN::fprop(){
        using namespace cuv;
        using namespace std;

        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];

        const matrix::value_type* srcData = p0.value.cdata().ptr();
        const matrix::value_type alpha = 1.;
  //    if(r0.can_overwrite_directly() || r0.can_add_directly()){
  //        const matrix::value_type beta = r0.can_add_directly() ? 1.0 : 0.0;
        if(CUDNN_VERSION != 1 && r0.can_overwrite_directly()) {
#if CUDNN_VERSION != 1
            const matrix::value_type beta = 0;
            matrix::value_type* destData = r0.overwrite_or_add_value()->ptr();
            CUDNN_CALL(cudnnPoolingForward(m_state->handle, m_state->poolingDesc, &alpha, m_state->imgDesc, srcData, &beta, m_state->outDesc, destData));
#else
            cuvAssert(false);
#endif
            if(p0.need_derivative)
            	m_result = r0.overwrite_or_add_value();  // save for bprop
        }
        else {
            const matrix::value_type beta = 0.;
            // reallocate *sigh*
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
			matrix::value_type* destData = v->ptr();
#if CUDNN_VERSION != 1
			CUDNN_CALL(cudnnPoolingForward(m_state->handle, m_state->poolingDesc, &alpha, m_state->imgDesc, srcData, &beta, m_state->outDesc, destData));
#else
			CUDNN_CALL(cudnnPoolingForward(m_state->handle, m_state->poolingDesc, m_state->imgDesc, srcData, m_state->outDesc, destData));
#endif
            r0.push(v);
            if(p0.need_derivative)
            	m_result = v; // save for bprop
        }

        if(!p0.need_derivative){
            p0.value.reset();
        }
    }

    void PoolingcuDNN::bprop(){
        using namespace cuv;
        using namespace std;

        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

		const matrix::value_type* srcData = m_result->ptr();
       	const matrix::value_type* srcDiffData = r0.delta.cdata().ptr();
		const matrix::value_type* destData = p0.value.cdata().ptr();
        const matrix::value_type alpha = 1.;

		if (CUDNN_VERSION != 1 && (p0.can_overwrite_directly() || p0.can_add_directly())) {
#if CUDNN_VERSION != 1
            const matrix::value_type beta = p0.can_add_directly() ? 1.0 : 0.0;
			matrix::value_type* destDiffData = p0.overwrite_or_add_value()->ptr();
			CUDNN_CALL(cudnnPoolingBackward(m_state->handle, m_state->poolingDesc, &alpha, m_state->outDesc, srcData, m_state->outDesc, srcDiffData, m_state->imgDesc, destData, &beta, m_state->imgDesc, destDiffData));
#else
            cuvAssert(false);
#endif
		} else {
            const matrix::value_type beta = 0.;
			value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
			matrix::value_type* destDiffData = v->ptr();
#if CUDNN_VERSION != 1
			CUDNN_CALL(cudnnPoolingBackward(m_state->handle, m_state->poolingDesc, &alpha, m_state->outDesc, srcData, m_state->outDesc, srcDiffData, m_state->imgDesc, destData, &beta, m_state->imgDesc, destDiffData));
#else
			CUDNN_CALL(cudnnPoolingBackward(m_state->handle, m_state->poolingDesc, m_state->outDesc, srcData, m_state->outDesc, srcDiffData, m_state->imgDesc, destData, m_state->imgDesc, destDiffData));
#endif
			p0.push(v);
		}

        p0.value.reset();
        r0.delta.reset();
        m_result.reset();
    }


    void PoolingcuDNN::_determine_shapes(){
    	using namespace std;
        /*
         * images    (nImages, nMaps, imgPixY, imgPixX)
         * out       (nImages, nMaps, nOutPixY, nOutPixX)
         */
        assert(m_params[0]->shape.size()==4);
        std::vector<unsigned int> img = m_params[0]->shape;
        std::vector<unsigned int> dst(4);

        dst[0] = img[0];
        dst[1] = img[1];

        //this doesn't work if imgPix=6, height=4, stride=2, outputs 1 when it should be 2
        //or if imgPix=4, height=4, stride=4, outputs 0 when it should be 1
        
#if 0
        dst[2] = DIVUP(img[2] + 2*m_vertical_pad - m_window_height, m_vertical_stride)+1;
        dst[3] = DIVUP(img[3] + 2*m_horizontal_pad - m_window_width, m_horizontal_stride)+1;
        bool defaultcase_y = m_vertical_stride == m_window_height && m_vertical_pad == 0 && (img[2] % m_window_height == 0);
        bool defaultcase_x = m_horizontal_stride == m_window_width && m_horizontal_pad == 0 && (img[3] % m_window_width == 0);
        if(m_vertical_stride != 1 && !defaultcase_y) dst[2] -= 1;
        if(m_horizontal_stride != 1 && !defaultcase_x) dst[3] -= 1;
#else
        dst[2] = DIVUP(img[2] + 2*m_vertical_pad - m_window_height + 1, m_vertical_stride);
        dst[3] = DIVUP(img[3] + 2*m_horizontal_pad - m_window_width + 1, m_horizontal_stride);
#endif

        m_results[0]->shape = dst;
        m_state.reset(new cudnn_pooling_state(this, m_params[0]->shape, m_results[0]->shape));

    }
}
