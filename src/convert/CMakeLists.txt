add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/out/convert
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/convert.py ${CMAKE_BINARY_DIR}/out/convert
    DEPENDS convert.py
    COMMENT "ln: out/convert -> convert.py"
)

add_custom_target(convert_link ALL 
    DEPENDS ${CMAKE_BINARY_DIR}/out/convert
    COMMENT "ln: out/convert -> convert.py"
)
