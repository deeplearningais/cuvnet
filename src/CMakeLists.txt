# ---------- Find Boost Headers/Libraries -----------------------
SET (Boost_FIND_REQUIRED TRUE)
SET (Boost_FIND_QUIETLY TRUE)
SET (Boost_USE_MULTITHREADED TRUE)
SET (Boost_USE_STATIC_LIBS FALSE)
SET (Boost_ADDITIONAL_VERSIONS "1.43" "1.44" "1.45" "1.46" "1.47" "1.48" )
SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMakeModules )
#SET(BOOST_ROOT "/usr/local/src/boost1.46-1.46.1")
#SET(Boost_NO_SYSTEM_PATHS TRUE)
FIND_PACKAGE( Boost 1.46 COMPONENTS date_time unit_test_framework serialization program_options iostreams filesystem signals thread system python REQUIRED )
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/local/cuda/include)
#INCLUDE_DIRECTORIES(~/checkout/git/mongo/include)
#INCLUDE_DIRECTORIES(/usr/local/cuv/include)

FIND_PACKAGE( LAPACK )
LINK_DIRECTORIES( ${LAPACK_LIBRARIES} )

FIND_PACKAGE( PythonLibs )

INCLUDE_DIRECTORIES($ENV{HOME}/pool/include/boost-numeric-bindings/)
#LINK_DIRECTORIES(/usr/local/cuv/lib)
LINK_DIRECTORIES(~/lib)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fno-strict-aliasing")


# this is for using tcmalloc, see http://code.google.com/p/gperftools/wiki/GooglePerformanceTools
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")

add_subdirectory(cuvnet)
add_subdirectory(third_party)
add_subdirectory(tools)
add_subdirectory(example EXCLUDE_FROM_ALL)
add_subdirectory(test EXCLUDE_FROM_ALL)
