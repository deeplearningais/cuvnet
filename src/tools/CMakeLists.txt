OPTION(USE_OPENCV "Use OpenCV if available" OFF)
IF(USE_OPENCV)
    FIND_PACKAGE( OpenCV )
    FIND_PACKAGE( tbb )
ENDIF(USE_OPENCV)


SET (CUVNET_TOOLS_SRC
    pca.cpp
    logging.cpp
    learner.cpp
    crossvalid.cpp
    read_amat.cpp
    orthonormalization.cpp)

INCLUDE_DIRECTORIES(/usr/local/src/magma_1.1.0/include)
LINK_DIRECTORIES(/usr/local/src/magma_1.1.0/lib)

IF (OpenCV_FOUND AND TBB_FOUND)
   LIST(APPEND CUVNET_TOOLS_SRC preprocess.cpp data_source.cpp)
ENDIF (OpenCV_FOUND AND TBB_FOUND)

add_library(cuvnet_tools STATIC ${CUVNET_TOOLS_SRC})

# MAGMA has circular dependencies on MAGMABLAS.
# We need to link with -lmagma -lmagmablas -lmagma
# CMake gets rid of double links, therefore we 
# introduce a dummy 2nd "magma2", which refers to
# the same file!!
#add_library(magma STATIC IMPORTED)
#set_property(TARGET magma PROPERTY
#            IMPORTED_LOCATION /usr/local/src/magma_1.1.0/lib/libmagma.a)
#add_library(magma2 STATIC IMPORTED)
#set_property(TARGET magma2 PROPERTY
#            IMPORTED_LOCATION /usr/local/src/magma_1.1.0/lib/libmagma.a)
#add_library(magmablas STATIC IMPORTED)
#set_property(TARGET magmablas PROPERTY
#            IMPORTED_LOCATION /usr/local/src/magma_1.1.0/lib/libmagmablas.a)

TARGET_LINK_LIBRARIES( cuvnet_tools ${OpenCV_LIBS} ${TBB_LIBS} ${LAPACK_LIBRARIES} npy mdbq zzip )

if ( OpenCV_FOUND  AND TBB_FOUND )
    TARGET_LINK_LIBRARIES( cuvnet_tools ${OpenCV_LIBS} ${TBB_LIBS} )
endif( OpenCV_FOUND  AND TBB_FOUND )

