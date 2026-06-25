#include "connect4_solver.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace {
constexpr int WIDTH = Connect4PerfectEngine::WIDTH;
constexpr int HEIGHT = Connect4PerfectEngine::HEIGHT;

uint64_t bottom_mask(int col) { return 1ULL << (col * (HEIGHT + 1)); }
uint64_t column_top(int col) { return 1ULL << (col * (HEIGHT + 1) + HEIGHT); }

constexpr std::array<int, 7> COLUMN_ORDER = {3, 2, 4, 1, 5, 0, 6};
const std::array<uint64_t, 7> BOTTOM = {
    bottom_mask(0), bottom_mask(1), bottom_mask(2), bottom_mask(3),
    bottom_mask(4), bottom_mask(5), bottom_mask(6),
};
const std::array<uint64_t, 7> TOP = {
    column_top(0), column_top(1), column_top(2), column_top(3),
    column_top(4), column_top(5), column_top(6),
};

int popcount(uint64_t x) { return __builtin_popcountll(x); }

uint64_t legal_mask(uint64_t mask) {
    uint64_t playable = 0;
    for (int col = 0; col < WIDTH; ++col) {
        if ((mask & TOP[col]) == 0)
            playable |= 1ULL << col;
    }
    return playable;
}

std::pair<uint64_t, uint64_t> play_col(uint64_t position, uint64_t mask,
                                       int col) {
    uint64_t new_mask = mask | (mask + BOTTOM[col]);
    uint64_t new_position = position ^ new_mask;
    return {new_position, new_mask};
}

bool has_won(uint64_t position) {
    uint64_t m = position;
    uint64_t r = m & (m >> (HEIGHT + 1));
    if (r & (r >> (2 * (HEIGHT + 1))))
        return true;
    r = m & (m >> 1);
    if (r & (r >> 2))
        return true;
    r = m & (m >> HEIGHT);
    if (r & (r >> (2 * HEIGHT)))
        return true;
    r = m & (m >> (HEIGHT + 2));
    if (r & (r >> (2 * (HEIGHT + 2))))
        return true;
    return false;
}

std::pair<uint64_t, uint64_t>
board_to_bitboards(const std::vector<std::vector<int>> &board) {
    uint64_t mask = 0;
    uint64_t position = 0;
    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            if (board[row][col] == 0)
                continue;
            // Engine row 0 = top, row HEIGHT-1 = bottom; bitboard row 0 = bottom.
            int bitrow = (HEIGHT - 1) - row;
            uint64_t bit = 1ULL << (col * (HEIGHT + 1) + bitrow);
            mask |= bit;
            if (board[row][col] == 1)
                position |= bit;
        }
    }
    return {position, mask};
}
} // namespace

int Connect4PerfectEngine::negamax(uint64_t position, uint64_t mask, int alpha,
                                   int beta) {
    Key key{position, mask};
    auto it = cache_.find(key);
    if (it != cache_.end())
        return it->second;

    uint64_t playable = legal_mask(mask);
    if (playable == 0) {
        cache_[key] = 0;
        return 0;
    }

    int max_score = -SCORE_MAX * 2;
    for (int col : COLUMN_ORDER) {
        if (((playable >> col) & 1ULL) == 0)
            continue;
        auto [child_pos, child_mask] = play_col(position, mask, col);
        if (has_won(child_pos)) {
            int score = SCORE_MAX - popcount(child_mask);
            cache_[key] = score;
            return score;
        }
        int score = -negamax(child_pos, child_mask, -beta, -alpha);
        max_score = std::max(max_score, score);
        alpha = std::max(alpha, score);
        if (alpha >= beta)
            break;
    }

    cache_[key] = max_score;
    return max_score;
}

int Connect4PerfectEngine::best_move(
    const std::vector<std::vector<int>> &board, int current_player) {
    auto [position, mask] = board_to_bitboards(board);
    if (current_player == -1)
        position ^= mask;

    // Solved opening: first player plays the centre column.
    if (mask == 0)
        return 3;

    uint64_t playable = legal_mask(mask);
    int best_col = COLUMN_ORDER[0];
    int best_score = -SCORE_MAX * 2;
    int alpha = -SCORE_MAX * 2;
    int beta = SCORE_MAX * 2;

    for (int col : COLUMN_ORDER) {
        if (((playable >> col) & 1ULL) == 0)
            continue;
        auto [child_pos, child_mask] = play_col(position, mask, col);
        if (has_won(child_pos))
            return col;
        int score = -negamax(child_pos, child_mask, -beta, -alpha);
        if (score > best_score) {
            best_score = score;
            best_col = col;
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta)
            break;
    }
    return best_col;
}

void Connect4PerfectEngine::clear_cache() { cache_.clear(); }
