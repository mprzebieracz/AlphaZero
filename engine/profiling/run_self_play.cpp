// Standalone self-play driver for profiling: runs N games of self-play against a
// TorchScript/TensorRT network and reports wall time and throughput. Optionally
// records a kineto (chrome trace) profile viewable in chrome://tracing / Perfetto.
#include "game/chess.hpp"
#include "game/connect4.hpp"
#include "self_play.hpp"
#include "utils/replay_buffer.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <torch/csrc/autograd/profiler_kineto.h>

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <connect4|chess> <network_path> <num_games> <thread_count>"
                     " [max_moves] [kineto_out] [mcts_simulations] [mcts_batch_size]\n"
                     "  Pass \"\" for kineto_out to skip trace recording.\n";
        return 1;
    }

    std::string game_name = argv[1];
    std::string network_path = argv[2];
    int num_games = std::stoi(argv[3]);
    int thread_count = std::stoi(argv[4]);
    int max_moves = (argc >= 6) ? std::stoi(argv[5]) : 512;
    std::string kineto_out = (argc >= 7) ? argv[6] : "";
    int mcts_simulations = (argc >= 8) ? std::stoi(argv[7]) : 800;
    int mcts_batch_size = (argc >= 9) ? std::stoi(argv[8]) : 32;

    std::shared_ptr<Game> initial_game;
    if (game_name == "connect4") {
        initial_game = std::make_shared<Connect4>();
    } else if (game_name == "chess") {
        initial_game = std::make_shared<Chess>();
    } else {
        std::cerr << "Unknown game: " << game_name << '\n';
        return 1;
    }

    if (!kineto_out.empty()) {
        torch::profiler::impl::ProfilerConfig config(torch::profiler::impl::ProfilerState::KINETO,
                                                     false, false, false, false, false);
        std::set<torch::profiler::impl::ActivityType> activities = {
            torch::profiler::impl::ActivityType::CPU, torch::profiler::impl::ActivityType::CUDA};
        torch::autograd::profiler::prepareProfiler(config, activities);
        torch::autograd::profiler::enableProfiler(config, activities);
    }

    ReplayBuffer replay_buffer(1000000);

    std::cout << "self-play: game=" << game_name << " games=" << num_games
              << " threads=" << thread_count << " sims=" << mcts_simulations
              << " batch=" << mcts_batch_size << " max_moves=" << max_moves << '\n';

    auto start = std::chrono::steady_clock::now();
    self_play(initial_game, network_path, replay_buffer, num_games, thread_count, mcts_simulations,
              mcts_batch_size, max_moves);
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    size_t moves = replay_buffer.get_size();
    std::cout << "elapsed: " << elapsed << " s\n"
              << "games/s: " << (num_games / elapsed) << '\n'
              << "moves:   " << moves << " (" << (moves / elapsed) << " moves/s, "
              << (moves * static_cast<size_t>(mcts_simulations) / elapsed) << " sims/s)\n";

    if (!kineto_out.empty()) {
        auto result = torch::autograd::profiler::disableProfiler();
        result->save(kineto_out);
        std::cout << "kineto trace saved to " << kineto_out << '\n';
    }
    return 0;
}
