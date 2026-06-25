#include "self_play.hpp"

#include "game/game.hpp"
#include "inference/basic_inferer.hpp"
#include "mcts.hpp"
#include "mcts/mcts_factory.hpp"
#include "replay_buffer.hpp"
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <memory>
#include <random>
#include <vector>

using std::string;

static torch::Tensor vector_to_tensor(std::vector<float> &data) {
    return torch::from_blob(data.data(), {static_cast<long>(data.size())},
                            torch::kFloat);
}

// trajectory[i] holds the state from the perspective of the player about to
// move at step i. The last recorded state is the eventual winner's, so it
// gets +terminal_reward; signs alternate going backwards in time.
void assign_trajectory_rewards(std::vector<Transition> &trajectory,
                               float terminal_reward) {
    float value = terminal_reward;
    for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
        it->reward = value;
        value = -value;
    }
}

static void play_game(std::unique_ptr<Game> game, std::unique_ptr<MCTS> mcts,
                      ReplayBuffer &replay_buffer) {
    game->reset();
    std::vector<Transition> trajectory;

    static thread_local std::mt19937 rng(std::random_device{}());

    while (!game->is_terminal()) {
        auto canonical_state = game->get_canonical_state();
        torch::Tensor game_state_tensor = std::move(canonical_state);
        game_state_tensor = game_state_tensor.unsqueeze(0);
        auto policy = mcts->search(*game);

        std::discrete_distribution<int> dist(policy.begin(), policy.end());
        int action = dist(rng);

        trajectory.emplace_back(game_state_tensor,
                                vector_to_tensor(policy).clone(), 0);
        game->step(action);
    }

    assign_trajectory_rewards(trajectory, game->reward());

    replay_buffer.add(trajectory);
}

void self_play(std::shared_ptr<Game> initial_game, string network_path,
               ReplayBuffer &replay_buffer, int num_games, int thread_count) {
    auto device = torch::Device(torch::cuda::is_available() ? "cuda" : "cpu");
    auto game = std::make_unique<Connect4>(device);

    auto inferer_factory = NetworkInfererFactory(network_path, device);
    MCTSFactory mcts_factory(inferer_factory);

    std::atomic<int> games_played{0};
    std::atomic<int> games_finished{0};
    std::mutex cout_mutex;

    auto worker = [&]() {
        while (true) {
            int current = games_played.fetch_add(1);
            if (current >= num_games)
                break;

            auto mcts = mcts_factory.get_mcts();
            play_game(game->clone(), std::move(mcts), replay_buffer);
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Games played: " << games_finished++ << "/"
                          << num_games << "\n";
            }
        }
    };

    // Use thread_count workers total: thread_count-1 extra threads + this one.
    std::vector<std::thread> threads;
    for (int i = 1; i < thread_count; ++i) {
        threads.emplace_back(worker);
    }
    worker();

    for (auto &t : threads) {
        t.join();
    }
}
