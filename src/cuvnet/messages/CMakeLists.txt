SET(ProtoFiles 
    gd.proto
    )
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})
ADD_LIBRARY(messages SHARED ${ProtoSources} ${ProtoHeaders})
