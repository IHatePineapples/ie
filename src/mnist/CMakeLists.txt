# Mostly from pytorch/examples/cpp/mnist/CMakeLists.txt
# The only difference, if that we need to export the model
# weights at the end of training.

set(examples_dir "${CMAKE_SOURCE_DIR}/extern/pytorch/examples")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/out)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${examples_dir}/cpp/tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/out/data/
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(mnist mnist.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/out/mnist
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<PATH:ABSOLUTE_PATH,mnist,${CMAKE_CURRENT_BINARY_DIR}> ${CMAKE_BINARY_DIR}/out/mnist
    DEPENDS mnist
    COMMENT "ln: out/mnist -> mnist"
)

add_custom_target(mnist_link ALL 
    DEPENDS ${CMAKE_BINARY_DIR}/out/mnist
    COMMENT "ln: out/mnist -> mnist"
)
