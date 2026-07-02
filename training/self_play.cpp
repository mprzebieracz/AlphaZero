#include "self_play.hpp"

#include "game/game.hpp"
#include "inference/basic_inferer.hpp"
#include "mcts.hpp"
#include "mcts/mcts_factory.hpp"
#include "replay_buffer.hpp"
#include <algorithm>
#include <atomic>
#include <c10/core/Device.h>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static torch::Tensor vector_to_tensor(std::vector<float> &data) {
    return torch::from_blob(data.data(), {static_cast<long>(data.size())}, torch::kFloat);
}

void assign_trajectory_rewards(std::vector<Transition> &trajectory, float terminal_reward) {
    float value = terminal_reward;
    for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
        it->reward = value;
        value = -value;
    }
}

static void play_game(std::shared_ptr<Game> game, MCTS &mcts, ReplayBuffer &replay_buffer,
                      int mcts_num_simulations, int mcts_batch_size, int max_moves) {
    game->reset();
    std::vector<Transition> trajectory;

    while (!game->is_terminal() && static_cast<int>(trajectory.size()) < max_moves) {
        auto shape = game->get_state_shape();
        torch::Tensor game_state_tensor = torch::empty(shape, torch::kFloat32);
        game->write_canonical_state(game_state_tensor.data_ptr<float>());
        game_state_tensor = game_state_tensor.unsqueeze(0);
        auto [policy, root_value] = mcts.search(*game, mcts_num_simulations, mcts_batch_size);
        (void)root_value;

        int action = -1;
        if (trajectory.size() < 30) {
            std::discrete_distribution<int> dist(policy.begin(), policy.end());
            static thread_local std::mt19937 rng(std::random_device{}());
            action = dist(rng);
        } else {
            action = static_cast<int>(
                std::distance(policy.begin(), std::max_element(policy.begin(), policy.end())));
        }

        trajectory.emplace_back(game_state_tensor, vector_to_tensor(policy).clone(), 0);
        game->step(action);
    }

    assign_trajectory_rewards(trajectory, game->reward());
    replay_buffer.add(trajectory);
}

void self_play(std::shared_ptr<Game> initial_game, std::string network_path,
               ReplayBuffer &replay_buffer, int num_games, int thread_count,
               int mcts_num_simulations, int mcts_batch_size, int max_moves) {
    auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");

    int wait_for_count = std::min(thread_count, 4) * mcts_batch_size;
    int timeout_ms = 2;
    auto inferer_factory = NetworkInfererFactory(network_path, device, wait_for_count, timeout_ms);
    MCTSFactory mcts_factory(inferer_factory);

    std::atomic<int> games_finished{0};

    std::vector<std::unique_ptr<MCTS>> thread_mcts;
    thread_mcts.reserve(thread_count);
    for (int t = 0; t < thread_count; t++) {
        thread_mcts.push_back(mcts_factory.get_mcts());
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(thread_count)
    for (int i = 0; i < num_games; i++) {
        auto &mcts = thread_mcts[omp_get_thread_num()];
        play_game(initial_game->clone(), *mcts, replay_buffer, mcts_num_simulations,
                  mcts_batch_size, max_moves);

        auto current_finished = ++games_finished;
        std::cout << "Games played: " << current_finished << "/" << num_games << '\n';
    }
#else
    for (int i = 0; i < num_games; i++) {
        auto &mcts = thread_mcts[i % thread_count];
        play_game(initial_game->clone(), *mcts, replay_buffer, mcts_num_simulations,
                  mcts_batch_size, max_moves);

        auto current_finished = ++games_finished;
        std::cout << "Games played: " << current_finished << "/" << num_games << '\n';
    }
#endif
}
