FILE(GLOB CUVNET_OPS_CPP ops/*.cpp)
add_subdirectory(tools)
add_library(cuvnet_ops  STATIC 
    ${CUVNET_OPS_CPP}
	op.cpp
	op_utils.cpp
	op_io.cpp
    derivative_test.cpp
	)
