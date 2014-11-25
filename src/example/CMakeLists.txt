#add_executable(hyperopt hyperopt.cpp )
link_directories(/usr/local/cuda/lib)

add_executable(lenet lenet.cpp)
add_executable(svm svm.cpp)
add_executable(linear_regression linear_regression.cpp)
add_executable(logistic_regression logistic_regression.cpp)

add_executable(minibatch_learning minibatch_learning.cpp)

SET(EXAMPLE_LIBS cuvnet cuv mdbq ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} pthread)

#target_link_libraries(hyperopt  ${EXAMPLE_LIBS})
target_link_libraries(linear_regression  ${EXAMPLE_LIBS})
#target_link_libraries(logistic_regression  ${EXAMPLE_LIBS})
#target_link_libraries(minibatch_learning  ${EXAMPLE_LIBS})
target_link_libraries(lenet  ${EXAMPLE_LIBS})
target_link_libraries(svm  ${EXAMPLE_LIBS})

if ( PYTHONLIBS_FOUND )
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    #add_executable(obj_detection obj_detection.cpp)
    #target_link_libraries(obj_detection  
        #${Boost_LIBRARIES} ${PYTHON_LIBRARIES} 

    PYTHON_ADD_MODULE(pylinreg SHARED linear_regression_python.cpp)

    SET_TARGET_PROPERTIES(pylinreg PROPERTIES OUTPUT_NAME "pylinreg")
    SET_TARGET_PROPERTIES(pylinreg PROPERTIES PREFIX "")
    TARGET_LINK_LIBRARIES(pylinreg 
        cuvnet cuv pthread jpeg log4cxx
        ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

    Add_Custom_Command(
        TARGET pylinreg
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pylinreg> ${CMAKE_BINARY_DIR})

    Add_Custom_Command(
        TARGET pylinreg
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/linear_regression_python.py ${CMAKE_BINARY_DIR})
endif( PYTHONLIBS_FOUND )
