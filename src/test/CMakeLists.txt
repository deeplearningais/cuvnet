# ---------- Find Boost Headers/Libraries -----------------------
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_TEST_DYN_LINK -DBOOST_TEST_NO_MAIN")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_TEST_DYN_LINK")

#find_package(GTest REQUIRED)
#CMAKE_FORCE_CXX_COMPILER(g++ GNU)
#include_directories(${GTEST_INCLUDE_DIRS})

set(PROJECT_TEST_NAME ${PROJECT_NAME_STR}_test)
include_directories(${CMAKE_BINARY_DIR}/src)

file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
target_link_libraries(${PROJECT_TEST_NAME}  ${GTEST_BOTH_LIBRARIES} ${Boost_LIBRARIES} cuvnet_datasets cuvnet_tools cuvnet_ops cuv jpeg npy pthread )

if ( PYTHONLIBS_FOUND )
    target_link_libraries(${PROJECT_TEST_NAME}  ${PYTHON_LIBRARIES})
endif( PYTHONLIBS_FOUND )

add_test(btest ${PROJECT_TEST_NAME})

