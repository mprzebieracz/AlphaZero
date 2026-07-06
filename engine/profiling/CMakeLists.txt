add_executable(run_self_play run_self_play.cpp ${CMAKE_SOURCE_DIR}/training/self_play.cpp)
target_link_libraries(run_self_play PRIVATE engine "${TORCH_LIBRARIES}")
target_include_directories(run_self_play PRIVATE
    ${CMAKE_SOURCE_DIR}/engine
    ${CMAKE_SOURCE_DIR}/training
)
target_compile_options(run_self_play PRIVATE -fopenmp)
target_link_options(run_self_play PRIVATE -fopenmp)
