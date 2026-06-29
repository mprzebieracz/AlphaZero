cmake_minimum_required(VERSION 3.18)

find_package(Torch REQUIRED)

set(TEST_TARGETS
    test_connect4
    test_replay_buffer
    test_mcts
    test_self_play_rewards
)

foreach(tgt IN LISTS TEST_TARGETS)
    add_executable(${tgt} ${tgt}.cpp)
    target_link_libraries(${tgt} PRIVATE engine "${TORCH_LIBRARIES}")
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}/engine
        ${CMAKE_SOURCE_DIR}/training
        ${TORCH_INCLUDE_DIRS}
    )
endforeach()

# assign_trajectory_rewards lives in training/, which isn't linked into the
# engine target — pull it in directly for the regression test.
target_sources(test_self_play_rewards PRIVATE
    ${CMAKE_SOURCE_DIR}/training/self_play.cpp
)

enable_testing()
foreach(tgt IN LISTS TEST_TARGETS)
    add_test(NAME ${tgt} COMMAND ${tgt})
endforeach()
