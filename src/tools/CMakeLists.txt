FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( tbb REQUIRED )

INCLUDE_DIRECTORIES(/usr/local/src/magma_1.1.0/include)
LINK_DIRECTORIES(/usr/local/src/magma_1.1.0/lib)

add_library(cuvnet_tools STATIC
    preprocess.cpp
    data_source.cpp
    logging.cpp
    learner.cpp
    crossvalid.cpp
    read_amat.cpp
    orthonormalization.cpp
    )

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

TARGET_LINK_LIBRARIES( cuvnet_tools ${OpenCV_LIBS} ${TBB_LIBS} ${LAPACK_LIBRARIES} mdbq zzip )
