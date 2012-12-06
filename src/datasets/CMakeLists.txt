add_library(cuvnet_datasets  STATIC 
	voc_detection.cpp
	amat_datasets.cpp
	read_amat.cpp
	bounding_box_tools.cpp
	)

target_link_libraries(cuvnet_datasets zzip)
