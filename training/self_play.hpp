#ifndef SELF_PLAY_HPP
#define SELF_PLAY_HPP

#include "game/game.hpp"
#include "replay_buffer.hpp"
#include <memory>
#include <string>
#include <thread>
#include <vector>

void self_play(std::shared_ptr<Game> game, std::string network_path, ReplayBuffer &replay_buf,
               int num_games = 100, int thread_count = std::thread::hardware_concurrency(),
               int mcts_num_simulations = 800, int mcts_batch_size = 32, int max_moves = 512);

void assign_trajectory_rewards(std::vector<Transition> &trajectory, float terminal_reward);

#endif // !SELF_PLAY_HPP
