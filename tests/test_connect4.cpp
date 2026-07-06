#include "game/connect4.hpp"
#include "test_common.hpp"
#include <cmath>
#include <torch/torch.h>

static void test_reset_state() {
    Connect4 g;
    CHECK_EQ(g.get_current_player(), 1);
    CHECK(!g.is_terminal());
    auto legal = g.get_legal_actions();
    CHECK_EQ(legal.size(), (size_t)Connect4::COLS);
    for (int c = 0; c < Connect4::COLS; ++c) CHECK_EQ(legal[c], c);
}

// reward() is expressed from the perspective of the player to move at the
// terminal state: the player is flipped even on a winning move, so the winner's
// opponent is "to move" and reward() is -1.
static void test_horizontal_win_player1() {
    Connect4 g;
    int seq[] = {0, 0, 1, 1, 2, 2};
    for (int a : seq) g.step(a);
    CHECK(!g.is_terminal());
    g.step(3);
    CHECK(g.is_terminal());
    CHECK_EQ(g.get_current_player(), -1);
    CHECK_NEAR(g.reward(), -1.0f, 1e-6);
}

static void test_vertical_win_player2() {
    Connect4 g;
    int seq[] = {0, 1, 0, 1, 0, 1, 2};
    for (int a : seq) g.step(a);
    CHECK(!g.is_terminal());
    g.step(1);
    CHECK(g.is_terminal());
    CHECK_EQ(g.get_current_player(), 1);
    CHECK_NEAR(g.reward(), -1.0f, 1e-6);
}

static void test_diagonal_win() {
    Connect4 g;
    int seq[] = {0, 1, 1, 2, 2, 3, 2, 3, 3};
    for (int a : seq) g.step(a);
    CHECK(!g.is_terminal());
    g.step(6);
    g.step(3);
    CHECK(g.is_terminal());
    CHECK_NEAR(g.reward(), -1.0f, 1e-6);
}

static void test_full_column_illegal() {
    Connect4 g;
    for (int i = 0; i < 6; ++i) {
        CHECK(!g.is_terminal());
        g.step(0);
    }
    for (int a : g.get_legal_actions()) CHECK(a != 0);
}

static void test_canonical_state_perspective() {
    Connect4 g;
    g.step(3);
    CHECK_EQ(g.get_current_player(), -1);
    auto shape = g.get_state_shape();
    torch::Tensor state = torch::empty(shape, torch::kFloat32);
    g.write_canonical_state(state.data_ptr<float>());
    CHECK_EQ(state.dim(), 3);
    CHECK_EQ(state.size(0), 1);
    CHECK_EQ(state.size(1), 6);
    CHECK_EQ(state.size(2), 7);
    CHECK_NEAR(state[0][5][3].item<float>(), -1.0f, 1e-6);
}

static void test_construct_from_board() {
    Connect4::board_t board = {};
    board[5][0] = 1;
    board[5][1] = -1;
    Connect4 g(board);
    CHECK_EQ(g.get_current_player(), 1);
    CHECK(!g.is_terminal());
}

int main() {
    RUN(test_reset_state);
    RUN(test_horizontal_win_player1);
    RUN(test_vertical_win_player2);
    RUN(test_diagonal_win);
    RUN(test_full_column_illegal);
    RUN(test_canonical_state_perspective);
    RUN(test_construct_from_board);
    std::printf("All Connect4 tests passed.\n");
    return 0;
}
