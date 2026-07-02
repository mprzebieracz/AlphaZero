#include "game/connect4.hpp"
#include "inference/inferer.hpp"
#include "mcts/mcts.hpp"
#include "test_common.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <torch/torch.h>

class UniformInferer : public Inferer {
  public:
    int action_dim;
    UniformInferer(int A) : Inferer(torch::Device(torch::kCPU)), action_dim(A) {}

    std::vector<inference_result>
    infer(const std::vector<const GameState *> &states) override {
        std::vector<inference_result> out;
        out.reserve(states.size());
        for (const auto *state : states) {
            auto legal = state->get_legal_actions();
            std::vector<float> logits(legal.size(), 0.0f);
            out.push_back(inference_result{legal, std::move(logits), 0.0f});
        }
        return out;
    }
};

class ColumnBiasedInferer : public Inferer {
  public:
    int action_dim;
    int marked_col;
    float bias;
    ColumnBiasedInferer(int A, int col, float b)
        : Inferer(torch::Device(torch::kCPU)), action_dim(A), marked_col(col), bias(b) {}

    std::vector<inference_result>
    infer(const std::vector<const GameState *> &states) override {
        std::vector<inference_result> out;
        out.reserve(states.size());
        for (const auto *state : states) {
            auto shape = state->get_state_shape();
            torch::Tensor tensor_state = torch::empty(shape, torch::kFloat32);
            state->write_canonical_state(tensor_state.data_ptr<float>());

            bool has_opp = false;
            for (int r = 0; r < tensor_state.size(1); ++r) {
                if (tensor_state[0][r][marked_col].item<float>() == -1.0f) {
                    has_opp = true;
                    break;
                }
            }
            auto legal = state->get_legal_actions();
            std::vector<float> logits(legal.size(), 0.0f);
            out.push_back(
                inference_result{legal, std::move(logits), has_opp ? bias : -bias});
        }
        return out;
    }
};

static void print_policy(const std::vector<float> &policy) {
    std::printf("       policy = [");
    for (size_t i = 0; i < policy.size(); ++i)
        std::printf("%s%.3f", i ? ", " : "", policy[i]);
    std::printf("]\n");
}

static void test_policy_is_valid_distribution() {
    Connect4 g;
    auto inf = std::make_unique<UniformInferer>(Connect4::action_dim);
    MCTS mcts(std::move(inf), 1.25f, 19652.0f, 0.0f, 0.3f);
    auto [policy, root_value] = mcts.search(g, 64, 4);
    (void)root_value;

    CHECK_EQ((int)policy.size(), Connect4::action_dim);
    float sum = std::accumulate(policy.begin(), policy.end(), 0.0f);
    CHECK_NEAR(sum, 1.0f, 1e-4);
    for (float p : policy) CHECK(p >= 0.0f);
}

static void play_setup_col3_is_decisive(Connect4 &g) {
    g.step(0);
    g.step(4);
    g.step(1);
    g.step(5);
    g.step(2);
    g.step(6);
    CHECK_EQ(g.get_current_player(), 1);
    CHECK(!g.is_terminal());
}

static void test_finds_immediate_one_ply_win() {
    Connect4 g;
    play_setup_col3_is_decisive(g);

    auto inf = std::make_unique<UniformInferer>(Connect4::action_dim);
    MCTS mcts(std::move(inf), 1.25f, 19652.0f, 0.0f, 0.3f);
    auto [policy, root_value] = mcts.search(g, 400, 8);
    (void)root_value;
    print_policy(policy);

    int best = (int)(std::max_element(policy.begin(), policy.end()) - policy.begin());
    CHECK_EQ(best, 3);
    CHECK(policy[3] > 0.5f);
}

static void test_search_does_not_pick_losing_move() {
    Connect4 g;
    play_setup_col3_is_decisive(g);

    auto inf = std::make_unique<UniformInferer>(Connect4::action_dim);
    MCTS mcts(std::move(inf), 1.25f, 19652.0f, 0.0f, 0.3f);
    auto [policy, root_value] = mcts.search(g, 600, 8);
    (void)root_value;

    for (int i = 0; i < Connect4::action_dim; ++i)
        if (i != 3) CHECK(policy[3] > policy[i]);
}

static void test_ucb_sign_root_avoids_child_good_for_opponent() {
    Connect4 g;
    auto inf = std::make_unique<ColumnBiasedInferer>(Connect4::action_dim, 0, 0.95f);
    MCTS mcts(std::move(inf), 1.25f, 19652.0f, 0.0f, 0.3f);
    auto [policy, root_value] = mcts.search(g, 200, 4);
    (void)root_value;
    print_policy(policy);

    int best = (int)(std::max_element(policy.begin(), policy.end()) - policy.begin());
    CHECK(best != 0);
    CHECK(policy[0] < 0.3f);
}

int main() {
    RUN(test_policy_is_valid_distribution);
    RUN(test_finds_immediate_one_ply_win);
    RUN(test_search_does_not_pick_losing_move);
    RUN(test_ucb_sign_root_avoids_child_good_for_opponent);
    std::printf("All MCTS tests passed.\n");
    return 0;
}
