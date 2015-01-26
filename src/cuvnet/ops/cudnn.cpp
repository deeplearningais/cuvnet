#include <boost/scope_exit.hpp>
#include <cuvnet/common.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/ops/cudnn.hpp>
#include <third_party/cudnn-6.5-linux-x64-R2-rc1/cudnn.h>

#define DIVUP(x,y) (((x)+ (y) -1) / (y))
#define CUDNN_CALL(XXX) if(1){ \
    cudnnStatus_t status = XXX; \
    if (status != CUDNN_STATUS_SUCCESS) \
        throw(std::string("ERROR ") + #XXX + ", status: " + boost::lexical_cast<std::string>(status));}
    
#define DESTROY(DESC) \
		CUDNN_CALL(cudnnDestroyTensorDescriptor(DESC));

#define DESTROY_FILTER(DESC) \
		CUDNN_CALL(cudnnDestroyFilterDescriptor(DESC));

#define CONSTRUCT(DESC) \
		CUDNN_CALL(cudnnCreateTensorDescriptor(&DESC));

#define CONSTRUCT_FILTER(DESC) \
		CUDNN_CALL(cudnnCreateFilterDescriptor(&DESC));


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
		cudnnHandle_t handle;
		cudnnTensorDescriptor_t imgDesc, outputDesc, biasDesc;
        cudnnFilterDescriptor_t filterDesc;
        cudnnConvolutionDescriptor_t convDesc;
        cudnnConvolutionFwdAlgo_t algo;
        cudnnAddMode_t add_mode;
        size_t workspace_size;
        cudnn_state(ConvolvecuDNN* conv,
                std::vector<unsigned int> img_shape, 
                std::vector<unsigned int> filter_shape,
                std::vector<unsigned int> out_shape,
                std::vector<unsigned int> bias_shape
                ){

            CUDNN_CALL(cudnnCreate(&handle));

            CONSTRUCT(imgDesc);
            CONSTRUCT(outputDesc);
            CONSTRUCT(biasDesc);
            CONSTRUCT_FILTER(filterDesc);

            cudnnDataType_t dtype = cudnn_data_type<matrix>();

            // Set descriptors
            CUDNN_CALL(cudnnSetTensor4dDescriptor(imgDesc, CUDNN_TENSOR_NCHW, dtype, img_shape[0], img_shape[1], img_shape[2], img_shape[3]));
            if(bias_shape.size())
                CUDNN_CALL(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, dtype, bias_shape[0], bias_shape[1], bias_shape[2], bias_shape[3]));
            CUDNN_CALL(cudnnSetFilter4dDescriptor(filterDesc, dtype, filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]));
            CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
            CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc, conv->m_padding_y, conv->m_padding_x, conv->m_ver_filt_stride, conv->m_hor_filt_stride, 1, 1, CUDNN_CONVOLUTION));

            if(0);
            else if(bias_shape.size() == 0);
            else if(bias_shape[2] == 1) add_mode = CUDNN_ADD_SAME_C;       // a (set of) 1x1 images
            else if(bias_shape[1] == 1) add_mode = CUDNN_ADD_FEATURE_MAP;  // a single image
            else add_mode = CUDNN_ADD_FULL_TENSOR;                         // same size as input

            // query output layout
            int n_out, c_out, h_out, w_out;
            CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(convDesc, imgDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));
            cuvAssert((unsigned)n_out == out_shape[0]);
            cuvAssert((unsigned)c_out == out_shape[1]);
            cuvAssert((unsigned)h_out == out_shape[2]);
            cuvAssert((unsigned)w_out == out_shape[3]);

            // Set and allocate output tensor descriptor
            CUDNN_CALL(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, dtype, n_out, c_out, h_out, w_out));

            CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
                    handle,
                    imgDesc,
                    filterDesc,
                    convDesc,
                    outputDesc,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    //CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                    0,
                    &algo));

            CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, imgDesc, filterDesc, convDesc, outputDesc, algo, &workspace_size));
        }
        ~cudnn_state(){
            DESTROY(imgDesc);
            DESTROY(outputDesc);
            DESTROY_FILTER(filterDesc);

            CUDNN_CALL(cudnnDestroyConvolutionDescriptor(convDesc));
            CUDNN_CALL(cudnnDestroy(handle));
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

        cuv::tensor<unsigned char,matrix::memory_space_type>  workspace(m_state->workspace_size, Op::value_ptr::s_allocator);

		const matrix::value_type alpha = 1.0;
        
        const matrix::value_type* imgData = p0.value.cdata().ptr();
        const matrix::value_type* filterData = p1.value.cdata().ptr();

		if (r0.can_overwrite_directly() || r0.can_add_directly()) {
            const matrix::value_type beta = r0.can_add_directly() ? 1.0 : 0.0;

			matrix::value_type* outputData = r0.overwrite_or_add_value()->ptr();
			// launch convolution on GPU
			CUDNN_CALL(cudnnConvolutionForward(m_state->handle, &alpha, 
                    m_state->imgDesc, imgData,
                    m_state->filterDesc, filterData,
                    m_state->convDesc, m_state->algo, workspace.ptr(), m_state->workspace_size,
                    &beta,
                    m_state->outputDesc, outputData));
            if(m_params.size() == 3){
                const float beta = 1.0;
                CUDNN_CALL(cudnnAddTensor(m_state->handle, m_state->add_mode, &alpha, m_state->biasDesc,
                            m_params[2]->value.cdata().ptr(), &beta,
                            m_state->outputDesc, outputData));
            }

            log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("convolve"));
            //LOG4CXX_WARN(log, "max  in: " << cuv::maximum(p0.value.cdata()));
            //LOG4CXX_WARN(log, "max out: " << cuv::maximum(*r0.overwrite_or_add_value()));

		} else {
			// reallocate *sigh*
            const matrix::value_type beta = 0.;
			value_ptr v(new value_type(r0.shape, cuvnet::get_global_allocator()));
			matrix::value_type* outputData = v->ptr();
			CUDNN_CALL(cudnnConvolutionForward(m_state->handle, &alpha, 
                    m_state->imgDesc, imgData,
                    m_state->filterDesc, filterData,
                    m_state->convDesc, m_state->algo, workspace.ptr(), m_state->workspace_size,
                    &beta,
                    m_state->outputDesc, outputData));
            if(m_params.size() == 3){
                const float beta = 1.0;
                CUDNN_CALL(cudnnAddTensor(m_state->handle, m_state->add_mode, &alpha, m_state->biasDesc,
                            m_params[2]->value.cdata().ptr(), &beta,
                            m_state->outputDesc, outputData));
            }
            log4cxx::LoggerPtr log(log4cxx::Logger::getLogger("convolve"));
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
				CUDNN_CALL(cudnnConvolutionBackwardFilter(m_state->handle, &alpha,
                        m_state->imgDesc, imgData,
                        m_state->outputDesc, diffData,
                        m_state->convDesc, &beta,
                        m_state->filterDesc, gradFilterData));
			} else {
                const matrix::value_type beta = 0.;
				value_ptr ptr(new value_type(p1.shape, value_ptr::s_allocator));

				matrix::value_type* gradFilterData = (*ptr).ptr();
				CUDNN_CALL(cudnnConvolutionBackwardFilter(m_state->handle, &alpha,
                        m_state->imgDesc, imgData,
                        m_state->outputDesc, diffData,
                        m_state->convDesc, &beta,
                        m_state->filterDesc, gradFilterData));
				p1.push(ptr);
			}
		}

        if(m_params.size() == 3 && m_params[2]->need_derivative){
            param_t::element_type& p2 = *m_params[2];

			const matrix::value_type* diffData = r0.delta.cdata().ptr();

            const matrix::value_type alpha = 1.0;
			if (p2.can_overwrite_directly() || p2.can_add_directly()) {
                const matrix::value_type beta = p2.can_add_directly() ? 1.0 : 0.0; 
				matrix::value_type* gradbiasData = (*p2.overwrite_or_add_value()).ptr();
				CUDNN_CALL(cudnnConvolutionBackwardBias(m_state->handle, &alpha,
                        m_state->outputDesc, diffData,
                        &beta,
                        m_state->biasDesc, gradbiasData));
			} else {
                const matrix::value_type beta = 0.;
				value_ptr ptr(new value_type(p2.shape, value_ptr::s_allocator));

				matrix::value_type* gradbiasData = (*ptr).ptr();
				CUDNN_CALL(cudnnConvolutionBackwardBias(m_state->handle, &alpha,
                        m_state->outputDesc, diffData,
                        &beta,
                        m_state->biasDesc, gradbiasData));
				p2.push(ptr);
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
				CUDNN_CALL(cudnnConvolutionBackwardData(m_state->handle, &alpha, m_state->filterDesc, filterData, m_state->outputDesc, diffData, m_state->convDesc, &beta, m_state->imgDesc, gradImgData));
			}
			else {
                const matrix::value_type beta = 0.;
				value_ptr ptr = p0.value;
				p0.value.reset();       // try to overwrite input activations
				value_type& v = ptr.data_onlyshape();

				matrix::value_type* gradImgData = v.ptr();

				CUDNN_CALL(cudnnConvolutionBackwardData(m_state->handle, &alpha, m_state->filterDesc, filterData, m_state->outputDesc, diffData, m_state->convDesc, &beta, m_state->imgDesc, gradImgData));

				p0.push(ptr);
			}
		}
	}

	void ConvolvecuDNN::_determine_shapes() {

		using namespace std;

		//dst       (nImg, nFilt, nModules, nModules)
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

		//unsigned int nOutPixX = nImgPixX + 2*m_padding_x - nFltPixX + 1;
		//unsigned int nOutPixY = nImgPixY + 2*m_padding_y - nFltPixY + 1;

        unsigned int nOutPixX = DIVUP(nImgPixX+2*m_padding_x-nFltPixX, m_hor_filt_stride)+1;
        unsigned int nOutPixY = DIVUP(nImgPixY+2*m_padding_y-nFltPixY, m_ver_filt_stride)+1;

        if(m_hor_filt_stride != 1) nOutPixX -= 1;
        if(m_ver_filt_stride != 1) nOutPixY -= 1;

		dst[0] = img[0];
		dst[1] = nFilt;
		dst[2] = nOutPixY;
		dst[3] = nOutPixX;
		m_results[0]->shape = dst;
        if(m_params.size() == 3)
            m_state.reset(new cudnn_state(this, m_params[0]->shape, m_params[1]->shape, m_results[0]->shape, m_params[2]->shape));
        else
            m_state.reset(new cudnn_state(this, m_params[0]->shape, m_params[1]->shape, m_results[0]->shape, std::vector<unsigned int>()));
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

    void PoolingcuDNN::release_data(){
    	m_result.reset();
        Op::release_data();
    }

    void PoolingcuDNN::fprop(){

        using namespace cuv;
        using namespace std;

        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        //unsigned int outy = p0.shape[1]/m_subsx;
    //    unsigned int outx = p0.shape[2]/m_subsx;

        //TODO: NCHW -> y then x


		cudnnStatus_t status;
		cudnnHandle_t handle;
		status = cudnnCreate(&handle);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR fprop cudnnCreate, status: " + boost::lexical_cast<std::string>(status));

		cudnnDataType_t dtype = cudnn_data_type<matrix>();

		cudnnPoolingDescriptor_t poolingDesc;
		status = cudnnCreatePoolingDescriptor(&poolingDesc);

		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR fprop cudnnCreatePoolingDescriptor, status: " + boost::lexical_cast<std::string>(status));

		status = cudnnSetPooling2dDescriptor(poolingDesc, m_mode == cuv::alex_conv::PT_MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE, m_window_height, m_window_width, m_vertical_pad, m_horizontal_pad, m_vertical_stride, m_horizontal_stride);

		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR fprop cudnnSetPoolingDescriptor, status: " + boost::lexical_cast<std::string>(status));

        cudnnTensorDescriptor_t srcDesc, destDesc;
        CONSTRUCT(srcDesc);
		status = cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dtype, p0.shape[0], p0.shape[1], p0.shape[2], p0.shape[3]);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR fprop cudnnSetTensor4dDescriptor(srcDesc), status: " + boost::lexical_cast<std::string>(status));

		const matrix::value_type* srcData = p0.value.cdata().ptr();

        CONSTRUCT(destDesc);
		status = cudnnSetTensor4dDescriptor(destDesc, CUDNN_TENSOR_NCHW, dtype, r0.shape[0], r0.shape[1], r0.shape[2], r0.shape[3]);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR fprop cudnnSetTensor4dDescriptor(destDesc), status: " + boost::lexical_cast<std::string>(status));

        const matrix::value_type alpha = 1.;
  //    if(r0.can_overwrite_directly() || r0.can_add_directly()){
  //        const matrix::value_type beta = r0.can_add_directly() ? 1.0 : 0.0;
        if(r0.can_overwrite_directly()) {
            const matrix::value_type beta = 0;
            matrix::value_type* destData = r0.overwrite_or_add_value()->ptr();

        	status = cudnnPoolingForward(handle, poolingDesc, &alpha, srcDesc, srcData, &beta, destDesc, destData);

    		if (status != CUDNN_STATUS_SUCCESS)
    			throw("ERROR fprop cudnnPoolingForward, status: " + boost::lexical_cast<std::string>(status));

            if(p0.need_derivative)
            	m_result = r0.overwrite_or_add_value();  // save for bprop
        }else{

            const matrix::value_type beta = 0.;
            // reallocate *sigh*
		//	value_ptr v(new value_type(r0.shape, cuvnet::get_global_allocator()));
            value_ptr v(new value_type(r0.shape, value_ptr::s_allocator));
			matrix::value_type* destData = v->ptr();
        	status = cudnnPoolingForward(handle, poolingDesc, &alpha, srcDesc, srcData, &beta, destDesc, destData);

        //		cout<<"result "<<endl<<(*r0.overwrite_or_add_value())[indices[0][0]]<<endl;
    		if (status != CUDNN_STATUS_SUCCESS)
    			throw("ERROR fprop cudnnPoolingForward, status: " + boost::lexical_cast<std::string>(status));

            r0.push(v);

            if(p0.need_derivative)
            	m_result = v; // save for bprop
        }

        DESTROY(srcDesc);
        DESTROY(destDesc);
        status = cudnnDestroyPoolingDescriptor(poolingDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnDestroyPoolingDescriptor(), status: " + boost::lexical_cast<std::string>(status));
		status = cudnnDestroy(handle);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnDestroy(), status: " + boost::lexical_cast<std::string>(status));

    }

    void PoolingcuDNN::bprop(){
        using namespace cuv;
        using namespace std;

        param_t::element_type&  p0 = *m_params[0];
        result_t::element_type& r0 = *m_results[0];
        assert(p0.need_derivative);

		cudnnStatus_t status;
		cudnnHandle_t handle;
		status = cudnnCreate(&handle);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnCreate, status: " + boost::lexical_cast<std::string>(status));

		cudnnDataType_t dtype = cudnn_data_type<matrix>();

		cudnnPoolingDescriptor_t poolingDesc;
		status = cudnnCreatePoolingDescriptor(&poolingDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnCreatePoolingDescriptor, status: " + boost::lexical_cast<std::string>(status));

		status = cudnnSetPooling2dDescriptor(poolingDesc, m_mode == cuv::alex_conv::PT_MAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE, m_window_height, m_window_width, m_vertical_pad, m_horizontal_pad, m_vertical_stride, m_horizontal_stride);

		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnSetPoolingDescriptor, status: " + boost::lexical_cast<std::string>(status));


        cudnnTensorDescriptor_t srcDesc, destDesc, srcDiffDesc, destDiffDesc;
        CONSTRUCT(srcDesc);
		status = cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, dtype, m_result.cdata().shape(0), m_result.cdata().shape(1), m_result.cdata().shape(2), m_result.cdata().shape(3));
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnSetTensor4dDescriptor(srcDesc), status: " + boost::lexical_cast<std::string>(status));

		const matrix::value_type* srcData = m_result->ptr();

        CONSTRUCT(srcDiffDesc);
		status = cudnnSetTensor4dDescriptor(srcDiffDesc, CUDNN_TENSOR_NCHW, dtype, r0.shape[0], r0.shape[1], r0.shape[2], r0.shape[3]);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnSetTensor4dDescriptor(srcDiffDesc), status: " + boost::lexical_cast<std::string>(status));

		const matrix::value_type* srcDiffData = r0.delta.cdata().ptr();

	//	cout<< "---r0 " << r0.shape[0] << " " << r0.shape[1] << " " << r0.shape[2] << " " << r0.shape[3]<< endl;
	//	cout << "---m_result " << m_result.cdata().shape(0)<< " " <<  m_result.cdata().shape(1)<< " " <<  m_result.cdata().shape(2)<< " " <<  m_result.cdata().shape(3) << endl;

        CONSTRUCT(destDesc);
		status = cudnnSetTensor4dDescriptor(destDesc, CUDNN_TENSOR_NCHW, dtype, p0.shape[0], p0.shape[1], p0.shape[2], p0.shape[3]);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnSetTensor4dDescriptor(destDesc), status: " + boost::lexical_cast<std::string>(status));

	//    matrix::value_type* destData = r0.overwrite_or_add_value()->ptr();
		const matrix::value_type* destData = p0.value.cdata().ptr();

        CONSTRUCT(destDiffDesc);
		status = cudnnSetTensor4dDescriptor(destDiffDesc, CUDNN_TENSOR_NCHW, dtype, p0.shape[0], p0.shape[1], p0.shape[2], p0.shape[3]);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnSetTensor4dDescriptor(destDiffDesc), status: " + boost::lexical_cast<std::string>(status));

        const matrix::value_type alpha = 1.;
		if (p0.can_overwrite_directly() || p0.can_add_directly()) {
            const matrix::value_type beta = p0.can_add_directly() ? 1.0 : 0.0;

			matrix::value_type* destDiffData = p0.overwrite_or_add_value()->ptr();

			status = cudnnPoolingBackward(handle, poolingDesc, &alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, &beta, destDiffDesc, destDiffData);
			if (status != CUDNN_STATUS_SUCCESS)
				throw("ERROR bprop cudnnPoolingBackward, status: " + boost::lexical_cast<std::string>(status));
		} else {

            const matrix::value_type beta = 0.;
			value_ptr v(new value_type(p0.shape, value_ptr::s_allocator));
			matrix::value_type* destDiffData = v->ptr();
			status = cudnnPoolingBackward(handle, poolingDesc, &alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, &beta, destDiffDesc, destDiffData);
			if (status != CUDNN_STATUS_SUCCESS)
				throw("ERROR bprop cudnnPoolingBackward, status: " + boost::lexical_cast<std::string>(status));
			p0.push(v);
		}

        p0.value.reset();
        r0.delta.reset();
        m_result.reset();

        status = cudnnDestroyPoolingDescriptor(poolingDesc);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnDestroyPoolingDescriptor(), status: " + boost::lexical_cast<std::string>(status));
        DESTROY(srcDesc);
        DESTROY(srcDiffDesc);
        DESTROY(destDesc);
        DESTROY(destDiffDesc);
		status = cudnnDestroy(handle);
		if (status != CUDNN_STATUS_SUCCESS)
			throw("ERROR bprop cudnnDestroy(), status: " + boost::lexical_cast<std::string>(status));

    }


    void PoolingcuDNN::_determine_shapes(){
        /*
         * images    (numImages, numChannels, imgPixY, imgPixX)
         * dst:      (numImages, numChannels, imgPixY, imgPixX)
         */
    	using namespace std;
        assert(m_params[0]->shape.size()==4);
        std::vector<unsigned int> img = m_params[0]->shape;
        std::vector<unsigned int> dst(4);

        dst[0] = img[0];
        dst[1] = img[1];
        //TODO: org code? + stride
        //dst[2] = img[2] / m_window_height;
        //dst[3] = img[3] / m_window_width;

        dst[2] = DIVUP(img[2] + 2*m_vertical_pad - m_window_height, m_vertical_stride)+1;
        dst[3] = DIVUP(img[3] + 2*m_horizontal_pad - m_window_width, m_horizontal_stride)+1;

        bool defaultcase_y = m_vertical_stride == m_window_height && m_vertical_pad == 0 && (img[2] % m_window_height == 0);
        bool defaultcase_x = m_horizontal_stride == m_window_width && m_horizontal_pad == 0 && (img[3] % m_window_width == 0);
        if(m_vertical_stride != 1 && !defaultcase_y) dst[2] -= 1;
        if(m_horizontal_stride != 1 && !defaultcase_x) dst[3] -= 1;

        m_results[0]->shape = dst;

    }
}
