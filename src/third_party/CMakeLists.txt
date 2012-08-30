INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/libnpy-0.5/include )

#libnpy-0.5/src/libname.c
add_library(npy STATIC
    libnpy-0.5/src/npy.c
    )
