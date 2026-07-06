#include "game/chess.hpp"
#include "test_common.hpp"

static Chess::board_t empty_board() {
    Chess::board_t b{};
    for (auto &row : b)
        row.fill(EMPTY);
    return b;
}

// Find the encoded action matching (r1,c1)->(r2,c2) [and promotion], or -1.
static int find_action(const Chess &game, int r1, int c1, int r2, int c2, int promotion = -1) {
    for (int act : game.get_legal_actions()) {
        ChessAction<> ca = Chess::decode_action(act);
        if (ca.r1 == r1 && ca.c1 == c1 && ca.r2 == r2 && ca.c2 == c2 &&
            (promotion == -1 || ca.promotion == promotion)) {
            return act;
        }
    }
    return -1;
}

static void test_initial_moves() {
    Chess game;
    CHECK_EQ(game.get_legal_actions().size(), (size_t)20);

    int e4 = find_action(game, 6, 4, 4, 4);
    CHECK(e4 != -1);
    game.step(e4);
    CHECK_EQ(game.get_board_state()[4][4], W_PAWN);
    CHECK_EQ(game.get_board_state()[6][4], EMPTY);
    CHECK_EQ(game.get_current_player(), 1);
}

static void test_castling_generated() {
    Chess game;
    auto b = game.get_board_state();
    b[7][5] = EMPTY;
    b[7][6] = EMPTY;
    game.set_custom_state(b, 0);

    CHECK(find_action(game, 7, 4, 7, 6) != -1);
}

static void test_castling_moves_rook() {
    Chess game;
    auto b = game.get_board_state();
    b[7][5] = EMPTY;
    b[7][6] = EMPTY;
    game.set_custom_state(b, 0);

    int castle = find_action(game, 7, 4, 7, 6);
    CHECK(castle != -1);
    game.step(castle);
    CHECK_EQ(game.get_board_state()[7][6], W_KING);
    CHECK_EQ(game.get_board_state()[7][5], W_ROOK);
    CHECK_EQ(game.get_board_state()[7][7], EMPTY);
    CHECK_EQ(game.get_board_state()[7][4], EMPTY);
}

static void test_no_castling_through_check() {
    auto b = empty_board();
    b[7][4] = W_KING;
    b[7][7] = W_ROOK;
    b[0][5] = B_ROOK; // attacks f1, the square the king passes through
    b[0][0] = B_KING;
    Chess game;
    game.set_custom_state(b, 0);

    CHECK_EQ(find_action(game, 7, 4, 7, 6), -1);
}

static void test_en_passant() {
    Chess game;
    auto b = game.get_board_state();
    b[6][4] = EMPTY;
    b[3][4] = W_PAWN;
    b[1][3] = B_PAWN;
    game.set_custom_state(b, 1);
    game.move_piece(1, 3, 3, 3); // black double push d7-d5

    int ep = find_action(game, 3, 4, 2, 3);
    CHECK(ep != -1);
    game.step(ep);
    CHECK_EQ(game.get_board_state()[3][3], EMPTY); // captured pawn removed
    CHECK_EQ(game.get_board_state()[2][3], W_PAWN);
}

static void test_promotion() {
    Chess game;
    auto b = game.get_board_state();
    b[1][0] = W_PAWN;
    b[0][0] = EMPTY;
    game.set_custom_state(b, 0);

    int promo = find_action(game, 1, 0, 0, 0, 1);
    CHECK(promo != -1);
    game.step(promo);
    CHECK_EQ(game.get_board_state()[0][0], W_QUEEN);
}

static void test_checkmate() {
    Chess game;
    game.move_piece(6, 5, 5, 5); // 1. f3
    game.move_piece(1, 4, 3, 4); // 1... e5
    game.move_piece(6, 6, 4, 6); // 2. g4
    game.move_piece(0, 3, 4, 7); // 2... Qh4#

    CHECK(game.is_terminal());
    CHECK_NEAR(game.reward(), -1.0f, 1e-6); // white (to move) is mated
}

static void test_stalemate() {
    auto b = empty_board();
    b[0][0] = B_KING;
    b[1][2] = W_QUEEN;
    b[2][2] = W_KING;
    Chess game;
    game.set_custom_state(b, 1, -1, 1, 1, 1);

    CHECK(game.is_terminal());
    CHECK_NEAR(game.reward(), 0.0f, 1e-6);
}

static void test_canonical_state() {
    Chess game;
    float buffer[19 * 64];
    game.write_canonical_state(buffer);

    // Plane 12: side-to-move (all ones for white).
    for (int i = 0; i < 64; ++i)
        CHECK_NEAR(buffer[12 * 64 + i], 1.0f, 1e-6);
    // Plane 0 (own pawns): white pawn at a2 -> row 6, col 0.
    CHECK_NEAR(buffer[0 * 64 + 6 * 8 + 0], 1.0f, 1e-6);
    // Plane 6 (opponent pawns): black pawn at a7 -> row 1, col 0.
    CHECK_NEAR(buffer[6 * 64 + 1 * 8 + 0], 1.0f, 1e-6);

    // From black's perspective the board is flipped and planes swap roles.
    game.set_custom_state(game.get_board_state(), 1);
    game.write_canonical_state(buffer);
    for (int i = 0; i < 64; ++i)
        CHECK_NEAR(buffer[12 * 64 + i], 0.0f, 1e-6);
    CHECK_NEAR(buffer[0 * 64 + 6 * 8 + 0], 1.0f, 1e-6);
    CHECK_NEAR(buffer[6 * 64 + 1 * 8 + 0], 1.0f, 1e-6);
}

static void test_king_in_check_restricts_moves() {
    auto b = empty_board();
    b[7][4] = W_KING;   // e1
    b[0][4] = B_ROOK;   // e8, checking
    b[6][3] = W_PAWN;   // d2
    b[7][6] = W_KNIGHT; // g1, can block on e2
    Chess game;
    game.set_custom_state(b, 0);

    auto actions = game.get_legal_actions();
    int valid_moves = 0;
    for (int act : actions) {
        ChessAction<> ca = Chess::decode_action(act);
        CHECK(ca.r1 != 6 || ca.c1 != 3); // pawn can't resolve the check
        if (ca.r1 == 7 && ca.c1 == 4) {  // king: d1, f1, f2 only
            CHECK((ca.r2 == 7 && ca.c2 == 3) || (ca.r2 == 7 && ca.c2 == 5) ||
                  (ca.r2 == 6 && ca.c2 == 5));
            valid_moves++;
        }
        if (ca.r1 == 7 && ca.c1 == 6) { // knight: only the e2 block
            CHECK(ca.r2 == 6 && ca.c2 == 4);
            valid_moves++;
        }
    }
    CHECK_EQ(valid_moves, 4);
    CHECK_EQ(actions.size(), (size_t)4);
}

static void test_pinned_piece_cannot_move() {
    auto b = empty_board();
    b[7][4] = W_KING;   // e1
    b[6][4] = W_KNIGHT; // e2, pinned
    b[0][4] = B_ROOK;   // e8, pinning
    b[6][3] = W_PAWN;   // d2, free
    Chess game;
    game.set_custom_state(b, 0);

    for (int act : game.get_legal_actions()) {
        ChessAction<> ca = Chess::decode_action(act);
        CHECK(!(ca.r1 == 6 && ca.c1 == 4)); // pinned knight must not move
    }
}

static void test_knight_check_must_be_resolved() {
    auto b = empty_board();
    b[7][4] = W_KING;   // e1
    b[5][3] = B_KNIGHT; // d3, checking
    b[7][7] = W_ROOK;   // h1
    b[6][2] = W_PAWN;   // c2, can capture the knight
    Chess game;
    game.set_custom_state(b, 0);

    int valid_moves = 0;
    for (int act : game.get_legal_actions()) {
        ChessAction<> ca = Chess::decode_action(act);
        CHECK(!(ca.r1 == 7 && ca.c1 == 7)); // rook can't resolve a knight check
        if (ca.r1 == 6 && ca.c1 == 2) {     // pawn: only the capture
            CHECK(ca.r2 == 5 && ca.c2 == 3);
            valid_moves++;
        }
        if (ca.r1 == 7 && ca.c1 == 4) // king escapes: d1, d2, e2, f1
            valid_moves++;
    }
    CHECK_EQ(valid_moves, 5);
}

static void test_bishop_pin_blocks_rook() {
    auto b = empty_board();
    b[7][7] = W_KING;   // h1
    b[6][6] = W_ROOK;   // g2, pinned diagonally
    b[0][0] = B_BISHOP; // a8
    Chess game;
    game.set_custom_state(b, 0);

    for (int act : game.get_legal_actions()) {
        ChessAction<> ca = Chess::decode_action(act);
        CHECK(!(ca.r1 == 6 && ca.c1 == 6)); // diagonally pinned rook can't move
    }
}

static void test_action_encoding_roundtrip() {
    for (int a : {0, 1, 12345, Chess::action_dim - 1}) {
        CHECK_EQ(Chess::encode_action(Chess::decode_action(a)), a);
    }
}

int main() {
    RUN(test_initial_moves);
    RUN(test_castling_generated);
    RUN(test_castling_moves_rook);
    RUN(test_no_castling_through_check);
    RUN(test_en_passant);
    RUN(test_promotion);
    RUN(test_checkmate);
    RUN(test_stalemate);
    RUN(test_canonical_state);
    RUN(test_king_in_check_restricts_moves);
    RUN(test_pinned_piece_cannot_move);
    RUN(test_knight_check_must_be_resolved);
    RUN(test_bishop_pin_blocks_rook);
    RUN(test_action_encoding_roundtrip);
    std::printf("All Chess tests passed.\n");
    return 0;
}
