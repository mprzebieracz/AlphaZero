// MCTS regression tests using a fake inferer (no TorchScript model needed).

#include "game/connect4.hpp"
#include "inference/inferer.hpp"
#include "mcts/mcts.hpp"
#include "test_common.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <torch/torch.h>

// Returns uniform logits and value 0 for every state.
class UniformInferer : public Inferer {
  public:
    int action_dim;
    UniformInferer(int A) : Inferer(torch::Device(torch::kCPU)), action_dim(A) {}

    std::vector<inference_result>
    infer(std::vector<GameState> states) override {
        std::vector<inference_result> out;
        out.reserve(states.size());
        for (size_t i = 0; i < states.size(); ++i) {
            torch::Tensor _s = std::move(states[i]);
            (void)_s;
            torch::Tensor logits = torch::zeros({action_dim}, torch::kFloat32);
            out.emplace_back(logits, 0.0f);
        }
        return out;
    }
};

// Returns +bias if `marked_col` contains an opponent piece, else -bias.
// Used to verify the UCB sign: a high inferer value at a child means that
// child's current player (= our opponent) is happy, so root must avoid it.
class ColumnBiasedInferer : public Inferer {
  public:
    int action_dim;
    int marked_col;
    float bias;
    ColumnBiasedInferer(int A, int col, float b)
        : Inferer(torch::Device(torch::kCPU)), action_dim(A), marked_col(col),
          bias(b) {}

    std::vector<inference_result>
    infer(std::vector<GameState> states) override {
        std::vector<inference_result> out;
        out.reserve(states.size());
        for (size_t i = 0; i < states.size(); ++i) {
            torch::Tensor s = std::move(states[i]);
            bool has_opp = false;
            for (int r = 0; r < s.size(1); ++r) {
                if (s[0][r][marked_col].item<float>() == -1.0f) {
                    has_opp = true;
                    break;
                }
            }
            torch::Tensor logits = torch::zeros({action_dim}, torch::kFloat32);
            out.emplace_back(logits, has_opp ? bias : -bias);
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
    auto policy = mcts.search(g, 64, 4, /*add_root_noise=*/false);

    CHECK_EQ((int)policy.size(), Connect4::action_dim);
    float sum = std::accumulate(policy.begin(), policy.end(), 0.0f);
    CHECK_NEAR(sum, 1.0f, 1e-4);
    for (float p : policy) CHECK(p >= 0.0f);
}

// P1 owns cols 0,1,2 on row 5; P2 owns cols 4,5,6. P1 to move. Col 3 wins
// for whoever plays it (P1 immediately, or P2 next turn if P1 doesn't take it).
static void play_setup_col3_is_decisive(Connect4 &g) {
    g.step(0); g.step(4);
    g.step(1); g.step(5);
    g.step(2); g.step(6);
    CHECK_EQ(g.get_current_player(), 1);
    CHECK(!g.is_terminal());
}

static void test_finds_immediate_one_ply_win() {
    Connect4 g;
    play_setup_col3_is_decisive(g);

    auto inf = std::make_unique<UniformInferer>(Connect4::action_dim);
    MCTS mcts(std::move(inf), 1.25f, 19652.0f, 0.0f, 0.3f);
    auto policy = mcts.search(g, 400, 8, /*add_root_noise=*/false);
    print_policy(policy);

    int best = (int)(std::max_element(policy.begin(), policy.end()) -
                     policy.begin());
    CHECK_EQ(best, 3);
    CHECK(policy[3] > 0.5f);
}

static void test_search_does_not_pick_losing_move() {
    // In this position, anything other than col 3 hands P2 the win.
    Connect4 g;
    play_setup_col3_is_decisive(g);

    auto inf = std::make_unique<UniformInferer>(Connect4::action_dim);
    MCTS mcts(std::move(inf), 1.25f, 19652.0f, 0.0f, 0.3f);
    auto policy = mcts.search(g, 600, 8, /*add_root_noise=*/false);

    for (int i = 0; i < Connect4::action_dim; ++i)
        if (i != 3) CHECK(policy[3] > policy[i]);
}

static void test_ucb_sign_root_avoids_child_good_for_opponent() {
    // Pre-fix bug: root picked the child with highest child.Q, i.e. the
    // child whose own player (the opponent) was happiest. Here that's
    // col 0 — with the buggy UCB sign root picks it ~94% of the time.
    Connect4 g;
    auto inf =
        std::make_unique<ColumnBiasedInferer>(Connect4::action_dim, 0, 0.95f);
    MCTS mcts(std::move(inf), 1.25f, 19652.0f, 0.0f, 0.3f);
    auto policy = mcts.search(g, 200, 4, /*add_root_noise=*/false);
    print_policy(policy);

    int best = (int)(std::max_element(policy.begin(), policy.end()) -
                     policy.begin());
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
