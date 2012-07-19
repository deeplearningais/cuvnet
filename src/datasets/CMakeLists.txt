add_library(cuvnet_datasets  STATIC 
	random_translation.cpp
	voc_detection.cpp
	)

target_link_libraries(cuvnet_datasets)
