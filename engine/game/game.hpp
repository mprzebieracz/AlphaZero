#ifndef GAME_HPP
#define GAME_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

// What the inference layer needs from a position: a canonical (side-to-move
// perspective) tensor encoding and the legal action indices.
class GameState {
  public:
    virtual ~GameState() = default;
    virtual void write_canonical_state(float *out_buffer) const = 0;
    virtual std::vector<int64_t> get_state_shape() const = 0;
    virtual std::vector<int> get_legal_actions() const = 0;
};

class Game : public GameState, public std::enable_shared_from_this<Game> {
  protected:
    Game(const Game &) = default;
    Game &operator=(const Game &) = default;

  public:
    Game() = default;
    virtual ~Game() = default;

    virtual void reset() = 0;

    // Size of the (fixed) action space; legal actions are a subset of [0, size).
    virtual int getActionSize() const = 0;

    // Apply a legal action. Always flips the side to move, including on a
    // game-ending move.
    virtual void step(int action) = 0;

    virtual bool is_terminal() const = 0;
    virtual int get_current_player() const = 0;

    // Value of a terminal state from the perspective of the player to move there
    // (0 when non-terminal). Because step() flips the player even on a winning
    // move, a decisive game always ends with reward() == -1: the player to move
    // is the one who was just mated / lost. MCTS's terminal backpropagation and
    // self-play's training-target assignment both rely on this convention.
    virtual float reward() const = 0;

    virtual std::shared_ptr<const GameState> get_canonical_state() const = 0;
    virtual std::shared_ptr<Game> clone() const = 0;
    virtual void render() const = 0;
};

template <int Rows, int Cols, typename CellT = int8_t>
class Game2D : public Game {
  public:
    static constexpr int ROWS = Rows;
    static constexpr int COLS = Cols;

    using cell_t = CellT;
    using row_t = std::array<cell_t, COLS>;
    using board_t = std::array<row_t, ROWS>;

    virtual board_t get_board_state() const = 0;
};

#endif // GAME_HPP
