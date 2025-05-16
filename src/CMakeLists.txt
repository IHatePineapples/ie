include_directories(AFTER
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sets some environment variables 
# needed in later subdirectories.
add_subdirectory(proto)

add_subdirectory(convert)
add_subdirectory(mnist)
add_subdirectory(parser)
add_subdirectory(proto)

