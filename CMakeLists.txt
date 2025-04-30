cmake_minimum_required(VERSION 3.12)

project(IE
    VERSION 0.0.1
    DESCRIPTION "Compile time integrated Inference Engine"
    HOMEPAGE_URL "https://github.com/IHatePineapples/ie"
    LANGUAGES C CXX
)

find_package(Torch REQUIRED)

# This is used to initialise
