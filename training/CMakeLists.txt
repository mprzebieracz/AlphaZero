cmake_minimum_required(VERSION 3.16)
project(self_play_bind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

find_package(pybind11 REQUIRED CONFIG)
find_package(Torch REQUIRED)

add_library(self_play_bind MODULE
    bindings.cpp
    self_play.cpp
)

set_target_properties(self_play_bind PROPERTIES PREFIX "")

target_link_libraries(self_play_bind PRIVATE
    pybind11::module
    engine
    torch
    torch_cpu
)
