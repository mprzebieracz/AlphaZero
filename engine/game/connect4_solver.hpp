#ifndef CONNECT4_SOLVER_HPP
#define CONNECT4_SOLVER_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

class Connect4PerfectEngine {
  public:
    static constexpr int WIDTH = 7;
    static constexpr int HEIGHT = 6;
    static constexpr int SCORE_MAX = WIDTH * HEIGHT + 1;

    int best_move(const std::vector<std::vector<int>> &board,
                  int current_player);

    void clear_cache();

  private:
    struct Key {
        uint64_t position;
        uint64_t mask;
        bool operator==(const Key &o) const {
            return position == o.position && mask == o.mask;
        }
    };

    struct KeyHash {
        size_t operator()(const Key &k) const {
            return std::hash<uint64_t>{}(k.position ^ (k.mask << 1));
        }
    };

    std::unordered_map<Key, int, KeyHash> cache_;

    int negamax(uint64_t position, uint64_t mask, int alpha, int beta);
};

#endif // CONNECT4_SOLVER_HPP
