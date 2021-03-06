#ifndef __CUVNET_TOOLS_NORMALIZATION_HPP__
#     define __CUVNET_TOOLS_NORMALIZATION_HPP__
#include <cuv/convolution_ops/convolution_ops.hpp>


namespace cuvnet
{
    /**
     * force weights to be in unit-ball on given axis.
     *
     * @param W weight matrix to be adjusted
     * @param axis the axis to be adjusted
     * @param thresh size of the ball
     * 
     * @return a tuple where 
     *         first value is the fraction of entries for which the constraint was violated (only works if not using alex_conv routine for projection),
     *         second value gives average norm to a hidden unit,
     */
    template<class M>
    std::pair<float,float>
    project_to_unit_ball(cuv::tensor<float, M>& W, int axis, float thresh){
        cuv::tensor<float, M> C = W;
        if(axis < 0)
            axis = W.ndim()-1;
        if(W.ndim() < 2)
            throw std::runtime_error("project_to_unit_ball: need >= 2-dimensional matrix, this one has " + boost::lexical_cast<std::string>(W.ndim()));
        if(axis >= W.ndim())
            throw std::runtime_error("project_to_unit_ball: axis parameter is not in matrix dimensions, of which there are " + boost::lexical_cast<std::string>(W.ndim()));

        unsigned int ax_shape = W.shape(axis);
        if(W.ndim() > 2){
            unsigned int other_shape = W.size() / ax_shape;
            if(axis == W.ndim()-1){
                axis = 1;
                C.reshape(cuv::extents[other_shape][ax_shape]);
            }else if(axis == 0){
                axis = 0;
                C.reshape(cuv::extents[ax_shape][other_shape]);
            }else{
                throw std::runtime_error("project_to_unit_ball: cannot normalize intermediate (0<axis<ndim-1) axis");
            }
        }
        //if(axis == 1 && (C.ndim() == 3 || C.ndim() == 4) && (C.shape(axis) % 16 == 0))
        //{
        //    cuv::alex_conv::project_to_ball(C, thresh);
        //    return std::make_pair(-1, 0.f);
        //}
        //ax_shape = C.shape(1-axis);
        cuv::tensor<float, M> ax(ax_shape);
        cuv::tensor<unsigned char, M> over_thresh(ax_shape);
        if(axis == 1){
            cuv::reduce_to_row(ax, C, cuv::RF_ADD_SQUARED);
        }
        else if(axis == 0){
            cuv::reduce_to_col(ax, C, cuv::RF_ADD_SQUARED);
        }

        float mean_squared_norm = cuv::mean(ax);

        cuv::apply_scalar_functor(over_thresh, ax, cuv::SF_GT, thresh * thresh);
        float frac_over_thresh = cuv::mean(over_thresh);
        if(frac_over_thresh == 0.f)
            return std::make_pair(0, std::sqrt(mean_squared_norm));
        cuv::apply_scalar_functor(ax, cuv::SF_SQRT); // ax[over_thresh] = sqrt(ax[over_thresh])
        cuv::apply_scalar_functor(ax, cuv::SF_MULT, 1.f / thresh);      // ax[over_thresh] *= 1/thresh
        over_thresh = !over_thresh;    // 
        cuv::apply_scalar_functor(ax, cuv::SF_MULT, 0.f, &over_thresh); // ax[!over_thresh] = 0
        cuv::apply_scalar_functor(ax, cuv::SF_ADD , 1.f, &over_thresh); // ax[!over_thresh] += 1
        if(axis == 1)
            cuv::matrix_divide_row(C, ax);
        else if(axis == 0)
            cuv::matrix_divide_col(C, ax);
        return std::make_pair(frac_over_thresh, std::sqrt(mean_squared_norm));
    }






    /**
     * force weights to be in unit-ball on given axis.
     *
     * @param W weight matrix to be adjusted
     * @param axis the axis to be adjusted
     * @param thresh size of the ball
     * @param logspace weights are in logspace => expf(w) i projected
     * @param surface if true, project to surface of ball, not inside ball
     */
    template<class M>
        void project_on_unit_ball(cuv::tensor<float, M>& W, int axis, float thresh, bool logspace, bool surface=false){
            cuv::tensor<float, M> C = W;

            if(axis < 0)
                axis = W.ndim()-1;
            if(W.ndim() < 2)
                throw std::runtime_error("project_to_unit_ball: need >2-dimensional matrix, this one has " + boost::lexical_cast<std::string>(W.ndim()));
            if(axis >= W.ndim())
                throw std::runtime_error("project_to_unit_ball: axis parameter is not in matrix dimensions, of which there are " + boost::lexical_cast<std::string>(W.ndim()));

            unsigned int ax_shape = W.shape(axis);
            if(W.ndim() > 2){
                unsigned int other_shape = W.size() / ax_shape;
                if(axis == W.ndim()-1){
                    axis = 1;
                    C.reshape(cuv::extents[other_shape][ax_shape]);
                }else if(axis == 0){
                    axis = 0;
                    C.reshape(cuv::extents[ax_shape][other_shape]);
                }else{
                    throw std::runtime_error("project_to_unit_ball: cannot normalize intermediate (0<axis<ndim-1) axis");
                }
            }
            //ax_shape = C.shape(1-axis);
            cuv::tensor<float, M> ax(ax_shape);
            if(logspace){
                if(axis == 1)     { cuv::reduce_to_row(ax, C, cuv::RF_MAX); }
                else if(axis == 0){ cuv::reduce_to_col(ax, C, cuv::RF_MAX); }

                cuv::apply_scalar_functor(ax, cuv::SF_NEGATE);
                cuv::matrix_op_vec(C, C, ax, axis, cuv::BF_ADD); 
                cuv::apply_scalar_functor(C, cuv::SF_EXP);
            }
            if(axis == 1)     { cuv::reduce_to_row(ax, C, cuv::RF_ADD_SQUARED); }
            else if(axis == 0){ cuv::reduce_to_col(ax, C, cuv::RF_ADD_SQUARED); }

            cuv::tensor<unsigned char, M> over_thresh(ax_shape);
            cuv::apply_scalar_functor(over_thresh, ax, cuv::SF_GT, thresh * thresh);
            cuv::apply_scalar_functor(ax, cuv::SF_SQRT);
            if(!surface){
                cuv::apply_scalar_functor(ax, cuv::SF_MULT, 1.f / thresh);      // ax[over_thresh] *= 1/thresh
                over_thresh = !over_thresh;    // 
                cuv::apply_scalar_functor(ax, cuv::SF_MULT, 0.f, &over_thresh); // ax[!over_thresh] = 0
                cuv::apply_scalar_functor(ax, cuv::SF_ADD , 1.f, &over_thresh); // ax[!over_thresh] += 1
            }else if(thresh != 1.f){
                cuv::apply_scalar_functor(ax, cuv::SF_MULT, 1.f / thresh); // ax[over_thresh] *= 1/thresh
            }

            if(axis == 1)     { cuv::matrix_divide_row(C, ax); }
            else if(axis == 0){ cuv::matrix_divide_col(C, ax); }

            if(logspace){
                cuv::apply_scalar_functor(C, cuv::SF_LOG);
            }
        }








    /**
     * force weight (-updates) to all have the same magnitude
     *
     * @param W weight matrix to be adjusted
     * @param axis the axis to be adjusted
     */
    template<class M>
    void project_to_mean(cuv::tensor<float, M>& W, int axis){
        cuv::tensor<float, M> C = W;
        if(axis < 0)
            axis = W.ndim()-1;
        if(W.ndim() < 2)
            throw std::runtime_error("project_to_unit_ball: need >2-dimensional matrix, this one has " + boost::lexical_cast<std::string>(W.ndim()));
        if(axis >= W.ndim())
            throw std::runtime_error("project_to_unit_ball: axis parameter is not in matrix dimensions, of which there are " + boost::lexical_cast<std::string>(W.ndim()));

        unsigned int ax_shape = W.shape(axis);
        if(W.ndim() > 2){
            unsigned int other_shape = W.size() / ax_shape;
            if(axis == W.ndim()-1){
                axis = 1;
                C.reshape(cuv::extents[other_shape][ax_shape]);
            }else if(axis == 0){
                axis = 0;
                C.reshape(cuv::extents[ax_shape][other_shape]);
            }else{
                throw std::runtime_error("project_to_unit_ball: cannot normalize intermediate (0<axis<ndim-1) axis");
            }
        }
        //ax_shape = C.shape(1-axis);
        cuv::tensor<float, M> ax(ax_shape);
        cuv::tensor<unsigned char, M> over_thresh(ax_shape);
        if(axis == 1){
            cuv::reduce_to_row(ax, C, cuv::RF_ADD_SQUARED);
        }
        else if(axis == 0){
            cuv::reduce_to_col(ax, C, cuv::RF_ADD_SQUARED);
        }
        float mean_sq_norm = cuv::mean(ax);

        cuv::apply_scalar_functor(ax, cuv::SF_SQRT); // ax = sqrt(ax)
        cuv::apply_scalar_functor(ax, cuv::SF_MULT, 1.f / std::sqrt(mean_sq_norm));
        if(axis == 1)
            cuv::matrix_divide_row(C, ax);
        else if(axis == 0)
            cuv::matrix_divide_col(C, ax);
    }
}
#endif /* __CUVNET_TOOLS_NORMALIZATION_HPP__ */
