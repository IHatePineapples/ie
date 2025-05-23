find_package(Protobuf REQUIRED)

# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx-ml.proto)
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO DLL_EXPORT onnx-ml.proto)
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS DESCRIPTORS PROTO_DESCS onnx-ml.proto)
add_library(onnx_proto onnx-ml.proto)
protobuf_generate(TARGET onnx_proto)

# add_library(onnx_proto STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(onnx_proto
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
                           PUBLIC ${Protobuf_INCLUDE_DIRS}
                           )
target_link_libraries(onnx_proto PRIVATE ${Protobuf_LIBRARIES} absl_strings)
