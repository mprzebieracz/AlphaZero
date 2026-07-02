find_package(Torch REQUIRED)
add_library(utils STATIC replay_buffer.cpp)
target_link_libraries(utils "${TORCH_LIBRARIES}")
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
