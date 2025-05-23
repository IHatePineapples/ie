add_executable(open open.cpp)

target_link_libraries(open onnx_proto ${Protobuf_LIBRARIES} absl_log_internal_check_op)
