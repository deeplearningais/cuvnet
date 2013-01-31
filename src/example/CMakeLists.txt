#add_executable(hyperopt hyperopt.cpp )
link_directories(/usr/local/cuda/lib)

#add_executable(linear_regression linear_regression.cpp)
#add_executable(logistic_regression logistic_regression.cpp)

#add_executable(minibatch_learning minibatch_learning.cpp)

SET(EXAMPLE_LIBS cuvnet cuv mdbq ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} pthread)

#target_link_libraries(hyperopt  ${EXAMPLE_LIBS})
#target_link_libraries(linear_regression  ${EXAMPLE_LIBS})
#target_link_libraries(logistic_regression  ${EXAMPLE_LIBS})
#target_link_libraries(minibatch_learning  ${EXAMPLE_LIBS})

if ( PYTHONLIBS_FOUND )
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    #add_executable(obj_detection obj_detection.cpp)
    #target_link_libraries(obj_detection  
        #${Boost_LIBRARIES} ${PYTHON_LIBRARIES} 
endif( PYTHONLIBS_FOUND )
