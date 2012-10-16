add_executable(hyperopt hyperopt.cpp )
add_executable(relational_auto_encoder relational_auto_encoder.cpp )
add_executable(auto_enc auto_enc.cpp )
add_executable(conv_auto_enc conv_auto_enc.cpp )
add_executable(logistic_regression logistic_regression.cpp )
add_executable(mlp mlp.cpp )
add_executable(lenet lenet.cpp )
add_executable(conv_nopool simple_conv_nopool.cpp )
add_executable(stacked_auto_enc3 stacked_auto_enc3.cpp)
#add_executable(relational_ae relational_ae.cpp)
#add_executable(descriptor_classification descriptor_classification.cpp)

#include(PrecompiledHeader)
#ADD_PRECOMPILED_HEADER(stacked_auto_enc3 stacked_auto_enc3.hpp)

#target_link_libraries(stacked_auto_enc2 ${Boost_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread)
#target_link_libraries(relational_ae ${Boost_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread)
#target_link_libraries(descriptor_classification ${Boost_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread)

FIND_PACKAGE( PythonLibs )
if ( PYTHONLIBS_FOUND )
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    add_executable(obj_detection obj_detection.cpp)
    target_link_libraries(hyperopt  ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools cuvnet_datasets pthread mdbq)
    target_link_libraries(obj_detection  
        ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} 
        cuv_convert cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuv_opt cuv_convolution_ops 
        cuvnet_tools cuvnet_datasets cuvnet_ops  
        png jpeg X11 npy pthread)
    target_link_libraries(relational_auto_encoder  ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools cuvnet_datasets pthread)
    target_link_libraries(stacked_auto_enc3  ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES} 
        cuvnet_tools cuvnet_datasets 
        cuv_convolution_ops cuv_cimg cuv_opt cuv_convert cuv_random cuv_matrix_ops cuv_tensor_ops cuv_basics cuv_tools cuvnet_ops 
        png npy pthread X11
        )
    target_link_libraries(logistic_regression ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread)
    target_link_libraries(lenet ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread cuv_convolution_ops cuv_convert)
    target_link_libraries(conv_nopool ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread cuv_convolution_ops cuv_convert)
    target_link_libraries(conv_auto_enc ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread) 
    target_link_libraries(auto_enc ${Boost_LIBRARIES}  ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread)
    target_link_libraries(mlp ${Boost_LIBRARIES}  ${PYTHON_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuv_cimg cuvnet_ops cuv_opt npy cuvnet_tools pthread)
endif( PYTHONLIBS_FOUND )
