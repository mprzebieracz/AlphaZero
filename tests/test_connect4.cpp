#include "game/connect4.hpp"
#include "test_common.hpp"
#include <cmath>

static void test_reset_state() {
    Connect4 g;
    CHECK_EQ(g.get_current_player(), 1);
    CHECK(!g.is_terminal());
    auto legal = g.get_legal_actions();
    CHECK_EQ(legal.size(), (size_t)Connect4::COLS);
    for (int c = 0; c < Connect4::COLS; ++c) CHECK_EQ(legal[c], c);
}

static void test_horizontal_win_player1() {
    // P1 plays cols 0,1,2,3 and wins on row 5.
    Connect4 g;
    int seq[] = {0, 0, 1, 1, 2, 2};
    for (int a : seq) g.step(a);
    CHECK(!g.is_terminal());
    g.step(3);
    CHECK(g.is_terminal());
    // Connect4 does not flip currentPlayer on terminal.
    CHECK_EQ(g.get_current_player(), 1);
    CHECK_NEAR(g.reward(), 1.0f, 1e-6);
}

static void test_vertical_win_player2() {
    Connect4 g;
    int seq[] = {0, 1, 0, 1, 0, 1, 2};
    for (int a : seq) g.step(a);
    CHECK(!g.is_terminal());
    g.step(1); // P2 completes vertical 4 in col 1.
    CHECK(g.is_terminal());
    CHECK_EQ(g.get_current_player(), -1);
    CHECK_NEAR(g.reward(), 1.0f, 1e-6);
}

static void test_diagonal_win() {
    // P1 owns the / diagonal (5,0),(4,1),(3,2),(2,3).
    Connect4 g;
    int seq[] = {0, 1, 1, 2, 2, 3, 2, 3, 3};
    for (int a : seq) g.step(a);
    CHECK(!g.is_terminal());
    g.step(6); // filler P2
    g.step(3); // P1 completes the diagonal.
    CHECK(g.is_terminal());
    CHECK_NEAR(g.reward(), 1.0f, 1e-6);
}

static void test_full_column_illegal() {
    // Six alternating plays in col 0 fills it with no 4-in-a-row.
    Connect4 g;
    for (int i = 0; i < 6; ++i) {
        CHECK(!g.is_terminal());
        g.step(0);
    }
    for (int a : g.get_legal_actions()) CHECK(a != 0);
}

static void test_canonical_state_perspective() {
    // P1 plays col 3. P2 now to move; the canonical view from P2 should
    // encode the opponent piece as -1.
    Connect4 g;
    g.step(3);
    CHECK_EQ(g.get_current_player(), -1);
    auto state = static_cast<torch::Tensor>(g.get_canonical_state());
    CHECK_EQ(state.dim(), 3);
    CHECK_EQ(state.size(0), 1);
    CHECK_EQ(state.size(1), 6);
    CHECK_EQ(state.size(2), 7);
    CHECK_NEAR(state[0][5][3].item<float>(), -1.0f, 1e-6);
}

int main() {
    RUN(test_reset_state);
    RUN(test_horizontal_win_player1);
    RUN(test_vertical_win_player2);
    RUN(test_diagonal_win);
    RUN(test_full_column_illegal);
    RUN(test_canonical_state_perspective);
    std::printf("All Connect4 tests passed.\n");
    return 0;
}
