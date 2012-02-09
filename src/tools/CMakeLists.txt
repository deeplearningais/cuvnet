FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( tbb REQUIRED )

add_library(cuvnet_tools
    preprocess.cpp
    data_source.cpp
    logging.cpp
    learner.cpp
    crossvalid.cpp
    read_amat.cpp
    )
TARGET_LINK_LIBRARIES( cuvnet_tools ${OpenCV_LIBS} ${TBB_LIBS} mdbq zzip)
