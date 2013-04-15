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
#INCLUDE_DIRECTORIES(~/checkout/git/mongo/include)
#INCLUDE_DIRECTORIES(/usr/local/cuv/include)

FIND_PACKAGE( LAPACK )
LINK_DIRECTORIES( ${LAPACK_LIBRARIES} )

FIND_PACKAGE( PythonLibs )


INCLUDE_DIRECTORIES(~/checkout/git/CUV/src)
LINK_DIRECTORIES(~/checkout/git/CUV/build/src/cuv)
LINK_DIRECTORIES(~/checkout/git/CUV/build/src/3rd_party/cuda_ndarray)


INCLUDE_DIRECTORIES($ENV{HOME}/pool/include/boost-numeric-bindings/)
#LINK_DIRECTORIES(/usr/local/cuv/lib)
LINK_DIRECTORIES(~/lib)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fno-strict-aliasing")

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/datasets/util/bbtools.jpg ${CMAKE_BINARY_DIR} COPYONLY)
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/datasets/util/lena.jpg ${CMAKE_BINARY_DIR} COPYONLY)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy-0.5/include )
OPTION(USE_OPENCV "Use OpenCV if available" OFF)
FIND_PACKAGE( tbb )

FILE(GLOB CUVNET_OPS_SRC cuvnet/ops/*.cpp)

SET(CUVNET_SRC 
    #datasets/voc_detection.cpp
    datasets/image_queue.cpp
	datasets/amat_datasets.cpp
	datasets/read_amat.cpp
	datasets/bounding_box_tools.cpp
    datasets/dataset_reader.cpp

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

    cuvnet/op.cpp
    cuvnet/op_utils.cpp
    cuvnet/op_io.cpp
    cuvnet/derivative_test.cpp

    third_party/libnpy-0.5/src/npy.c

    ${CUVNET_OPS_SRC}
    )


if ( PYTHONLIBS_FOUND )
    include_directories( ${PYTHON_INCLUDE_DIRS} )
    LIST(APPEND CUVNET_SRC cuvnet/tools/python_helper.cpp)
elseif( PYTHONLIBS_FOUND )
    SET(PYTHON_LIBRARIES "")
endif( PYTHONLIBS_FOUND )

# this is for using tcmalloc, see http://code.google.com/p/gperftools/wiki/GooglePerformanceTools
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

add_library(cuvnet SHARED ${CUVNET_SRC})

TARGET_LINK_LIBRARIES( cuvnet zzip mdbq zzip log4cxx cuv tp_theano ${OpenCV_LIBS} ${TBB_LIBS} ${LAPACK_LIBRARIES} )

#add_subdirectory(cuvnet)
#add_subdirectory(third_party)
add_subdirectory(example EXCLUDE_FROM_ALL)
add_subdirectory(test EXCLUDE_FROM_ALL)
