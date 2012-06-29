# ---------- Find Boost Headers/Libraries -----------------------
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

find_package(GTest REQUIRED)
#CMAKE_FORCE_CXX_COMPILER(g++ GNU)

set(PROJECT_TEST_NAME ${PROJECT_NAME_STR}_test)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/src)

file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
target_link_libraries(${PROJECT_TEST_NAME}  ${GTEST_BOTH_LIBRARIES} ${Boost_LIBRARIES} cuv cuv_tools cuv_basics cuv_tensor_ops cuv_matrix_ops cuv_random cuvnet_ops cuvnet_datasets  cuvnet_tools cuv_opt npy pthread )

if ( PYTHONLIBS_FOUND )
    target_link_libraries(${PROJECT_TEST_NAME}  ${PYTHON_LIBRARIES})
endif( PYTHONLIBS_FOUND )

add_test(gtest ${PROJECT_TEST_NAME})

