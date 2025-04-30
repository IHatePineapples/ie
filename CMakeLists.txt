cmake_minimum_required(VERSION 3.12)

project(IE
    VERSION 0.0.1
    DESCRIPTION "Compile time integrated Inference Engine"
    HOMEPAGE_URL "https://github.com/IHatePineapples/ie"
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/out)

add_subdirectory(src)
