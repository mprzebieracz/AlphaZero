// Regression test for the value-label sign for self-play trajectories.
// trajectory.back() (the eventual winner's last state) must get
// +terminal_reward; signs alternate going backwards.

#include "self_play.hpp"
#include "test_common.hpp"
#include "utils/replay_buffer.hpp"
#include <torch/torch.h>

static Transition zero_transition() {
    auto s = torch::zeros({1, 6, 7}, torch::kFloat32);
    auto p = torch::zeros({7}, torch::kFloat32);
    return Transition(s, p, 0.0f);
}

static void test_odd_length_game_player1_wins() {
    // 7-move game, P1 wins. traj[6] is P1's last state -> +1, alternating.
    std::vector<Transition> traj;
    for (int i = 0; i < 7; ++i) traj.push_back(zero_transition());

    assign_trajectory_rewards(traj, 1.0f);

    CHECK_NEAR(traj[6].reward, 1.0f, 1e-6);
    CHECK_NEAR(traj[5].reward, -1.0f, 1e-6);
    CHECK_NEAR(traj[4].reward, 1.0f, 1e-6);
    CHECK_NEAR(traj[3].reward, -1.0f, 1e-6);
    CHECK_NEAR(traj[2].reward, 1.0f, 1e-6);
    CHECK_NEAR(traj[1].reward, -1.0f, 1e-6);
    CHECK_NEAR(traj[0].reward, 1.0f, 1e-6);
}

static void test_even_length_game_player2_wins() {
    // 8-move game, P2 wins. traj[7] is P2's last state -> +1.
    // The pre-fix bug flipped every sign for even-length games.
    std::vector<Transition> traj;
    for (int i = 0; i < 8; ++i) traj.push_back(zero_transition());

    assign_trajectory_rewards(traj, 1.0f);

    CHECK_NEAR(traj[7].reward, 1.0f, 1e-6);
    CHECK_NEAR(traj[6].reward, -1.0f, 1e-6);
    CHECK_NEAR(traj[5].reward, 1.0f, 1e-6);
    CHECK_NEAR(traj[4].reward, -1.0f, 1e-6);
    CHECK_NEAR(traj[3].reward, 1.0f, 1e-6);
    CHECK_NEAR(traj[2].reward, -1.0f, 1e-6);
    CHECK_NEAR(traj[1].reward, 1.0f, 1e-6);
    CHECK_NEAR(traj[0].reward, -1.0f, 1e-6);
}

static void test_draw_assigns_all_zero() {
    std::vector<Transition> traj;
    for (int i = 0; i < 42; ++i) traj.push_back(zero_transition());
    assign_trajectory_rewards(traj, 0.0f);
    for (const auto &t : traj) CHECK_NEAR(t.reward, 0.0f, 1e-6);
}

static void test_empty_trajectory() {
    std::vector<Transition> traj;
    assign_trajectory_rewards(traj, 1.0f); // must not crash
    CHECK_EQ(traj.size(), (size_t)0);
}

static void test_single_move_trajectory() {
    std::vector<Transition> traj;
    traj.push_back(zero_transition());
    assign_trajectory_rewards(traj, 1.0f);
    CHECK_NEAR(traj[0].reward, 1.0f, 1e-6);
}

int main() {
    RUN(test_odd_length_game_player1_wins);
    RUN(test_even_length_game_player2_wins);
    RUN(test_draw_assigns_all_zero);
    RUN(test_empty_trajectory);
    RUN(test_single_move_trajectory);
    std::printf("All self-play reward tests passed.\n");
    return 0;
}
