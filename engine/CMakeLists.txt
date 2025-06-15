cmake_minimum_required(VERSION 3.16)
project(engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# All source files
file(GLOB_RECURSE SOURCES
    game/*.cpp
    inference/*.cpp
    mcts/*.cpp
)
list(FILTER SOURCES EXCLUDE REGEX "bindings.cpp")


find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})
find_package(pybind11 REQUIRED CONFIG)

find_package(Torch REQUIRED)

add_library(engine STATIC ${SOURCES})
target_compile_definitions(engine PUBLIC _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(engine PUBLIC "${TORCH_LIBRARIES}")
target_include_directories(engine PUBLIC ${TORCH_INCLUDE_DIRS})

# Add include directories so you can #include "mcts/mcts.hpp" etc.
target_include_directories(engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/game
    ${CMAKE_CURRENT_SOURCE_DIR}/mcts
)

add_library(engine_bind MODULE
  bindings.cpp
  ${SOURCES}
)

set_target_properties(engine_bind PROPERTIES PREFIX "")

target_link_libraries(engine_bind PRIVATE
    pybind11::module
    engine
    "${TORCH_LIBRARIES}"
)



