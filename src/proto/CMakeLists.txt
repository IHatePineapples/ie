find_package(Protobuf REQUIRED)

# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx-ml.proto)
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO DLL_EXPORT onnx-ml.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS DESCRIPTORS PROTO_DESCS onnx-ml.proto)

add_library(onnx_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(onnx_proto
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
                           INTERFACE ${Protobuf_INCLUDE_DIRS}
                           )
target_link_libraries(onnx_proto ${Protobuf_LIBRARIES})
