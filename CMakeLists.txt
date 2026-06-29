cmake_minimum_required(VERSION 3.18)
project(my_ai_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
  set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "")
endif()

set(CMAKE_BUILD_RPATH "$ENV{HOME}/libs/libtorch/lib")

add_subdirectory(engine)
add_subdirectory(training)

option(BUILD_TESTS "Build C++ tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TORCH_CUDA_FOUND)
  message(STATUS "Compiling with CUDA support")
else()
  message(STATUS "Compiling without CUDA support")
endif()

