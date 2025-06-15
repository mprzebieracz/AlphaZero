#ifndef GAME_HPP
#define GAME_HPP

#include <memory>
#include <torch/torch.h>
#include <vector>

// Abstract base class for game states, enabling hashing and comparison if
// needed
class GameState {
  public:
    virtual ~GameState() = default;

    // Return the canonical tensor representation: (Channels, Height, Width)
    virtual torch::Tensor getCanonicalState() const = 0;

    // Equality comparison for hashing or state lookup
    virtual bool operator==(const GameState &other) const = 0;

    // Provide a hash value (if using unordered containers)
    virtual std::size_t hash() const = 0;
};

// Abstract base class for Games
class Game {
  public:
    virtual ~Game() = default;

    // Reset game to initial state
    virtual void reset() = 0;

    // Return the dimension of the action space (number of possible discrete
    // actions)
    virtual int getActionSize() const = 0;

    // Return a list of legal action indices in the current state
    virtual std::vector<int> getLegalActions() const = 0;

    // Apply the given action, modifying the game state
    virtual void step(int action) = 0;

    // Check if the current state is terminal (game over)
    virtual bool is_terminal() const = 0;

    // Compute and return the reward for the current state (from perspective of
    // current player)
    virtual float reward() const = 0;

    // Get the canonical representation of the state for neural network input
    virtual torch::Tensor get_canonical_state() const = 0;

    // Produce a deep copy of the game (for tree search branching)
    virtual std::unique_ptr<Game> clone() const = 0;

    // Optional: render the current state (e.g., for debugging/visualization)
    virtual void render() const = 0;
};

#endif // GAME_HPP
