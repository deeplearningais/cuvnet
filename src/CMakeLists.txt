# ---------- Find Boost Headers/Libraries -----------------------
SET (Boost_FIND_REQUIRED TRUE)
SET (Boost_FIND_QUIETLY TRUE)
SET (Boost_USE_MULTITHREADED TRUE)
SET (Boost_USE_STATIC_LIBS FALSE)
SET (Boost_ADDITIONAL_VERSIONS "1.43" "1.44" "1.45" "1.46" "1.47" "1.48" "1.49")
SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMakeModules )
#SET(BOOST_ROOT "/usr/local/src/boost1.46-1.46.1")
#SET(Boost_NO_SYSTEM_PATHS TRUE)
FIND_PACKAGE( Boost 1.46 COMPONENTS date_time filesystem unit_test_framework serialization program_options iostreams signals thread system python REQUIRED )
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/local/cuda/include)
SET(CUV_BASE_DIRECTORY "/home/local/cuv" CACHE STRING "Checkout directory for cuv")
SET(CUV_BUILD_TYPE "debug" CACHE STRING "subdirectory of CUV_BASE_DIRECTORY containing build")
SET(CUV_INCLUDE_DIR "${CUV_BASE_DIRECTORY}/src")
SET(CUV_LIB_DIR "${CUV_BASE_DIRECTORY}/${CUV_BUILD_TYPE}/src/cuv"
                "${CUV_BASE_DIRECTORY}/${CUV_BUILD_TYPE}/src/3rd_party/cuda_ndarray"
                "${CUV_BASE_DIRECTORY}/${CUV_BUILD_TYPE}/src/cuv_python"
                "/usr/lib/python2.7/dist-packages/cuv_python"
                )
SET(CUV_LIBRARY cuv tp_theano)
SET(CUV_PYTHON_LIBRARY _cuv_python.so)

INCLUDE_DIRECTORIES(${CUV_INCLUDE_DIR})
LINK_DIRECTORIES(${CUV_LIB_DIR})

FIND_PACKAGE( LAPACK )
FIND_PACKAGE( CUDA )
LINK_DIRECTORIES( ${LAPACK_LIBRARIES} )

FIND_PACKAGE( PythonLibs 2.7 REQUIRED )

FIND_PACKAGE( Protobuf  REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})


INCLUDE_DIRECTORIES($ENV{HOME}/pool/include/boost-numeric-bindings/)
LINK_DIRECTORIES(~/lib)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fno-strict-aliasing")

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/datasets/util/bbtools.jpg ${CMAKE_BINARY_DIR} COPYONLY)
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/datasets/util/lena.jpg ${CMAKE_BINARY_DIR} COPYONLY)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy-0.5/include )
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ )

OPTION(USE_OPENCV "Use OpenCV if available" OFF)
OPTION(USE_THEANO_WRAPPERS "Use Theano wrappers (requires CUV to be compiled with python support" ON)
IF(USE_THEANO_WRAPPERS)
ELSE(USE_THEANO_WRAPPERS)
    ADD_DEFINITIONS(-DNO_THEANO_WRAPPERS=1)
ENDIF(USE_THEANO_WRAPPERS)

FIND_PACKAGE( tbb )
FIND_PACKAGE( OpenCV REQUIRED )

FILE(GLOB CUVNET_OPS_SRC cuvnet/ops/*.cpp)
add_subdirectory(cuvnet/messages)

SET(CUVNET_SRC 
    #datasets/voc_detection.cpp
    datasets/image_queue.cpp
	datasets/amat_datasets.cpp
	datasets/npy_datasets.cpp
	datasets/read_amat.cpp
	datasets/bounding_box_tools.cpp
    datasets/dataset_reader.cpp

    cuvnet/smart_ptr.cpp
    cuvnet/tools/pca.cpp
    cuvnet/tools/logging.cpp
    cuvnet/tools/monitor.cpp
    cuvnet/tools/learner.cpp
    cuvnet/tools/crossvalid.cpp
    cuvnet/tools/gradient_descent.cpp
    cuvnet/tools/orthonormalization.cpp
    cuvnet/tools/network_communication.cpp
    cuvnet/tools/dataset_dumper.cpp
    cuvnet/tools/dumper.cpp
    cuvnet/tools/learner2.cpp
    cuvnet/tools/simple_dataset_learner.cpp

    cuvnet/op.cpp
    cuvnet/op_utils.cpp
    cuvnet/op_io.cpp
    cuvnet/derivative_test.cpp
    cuvnet/graph_modifiers.cpp

    cuvnet/models/models.cpp
    cuvnet/models/initialization.cpp
    cuvnet/models/logistic_regression.cpp
    cuvnet/models/linear_regression.cpp
    cuvnet/models/mlp.cpp
    cuvnet/models/conv_layer.cpp

    third_party/libnpy-0.5/src/npy.c
    third_party/cnpy/cnpy.cpp

    ${CUVNET_OPS_SRC}
    )

INSTALL(DIRECTORY ./
    DESTINATION include/cuvnet
    FILES_MATCHING PATTERN "*.hpp"
    )

if ( PYTHONLIBS_FOUND )
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    LIST(APPEND CUVNET_SRC cuvnet/tools/python_helper.cpp)
    add_subdirectory(python/cuvnet)
elseif( PYTHONLIBS_FOUND )
    SET(PYTHON_LIBRARIES "")
endif( PYTHONLIBS_FOUND )

# this is for using tcmalloc, see http://code.google.com/p/gperftools/wiki/GooglePerformanceTools
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

add_library(cuvnet SHARED ${CUVNET_SRC})

TARGET_LINK_LIBRARIES( cuvnet zzip mdbq zzip log4cxx ${OpenCV_LIBS} ${TBB_LIBS} ${LAPACK_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES}  ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} ${PROTOBUF_LIBRARIES} messages)

#add_subdirectory(cuvnet)
#add_subdirectory(third_party)
add_subdirectory(example EXCLUDE_FROM_ALL)
add_subdirectory(test EXCLUDE_FROM_ALL)

INSTALL(TARGETS cuvnet
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
